# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict, List, Union
import random
import numpy as np
import sapien
import torch

from magma_core.simulation.envs import DefaultEnv

from mani_skill.utils.structs import Pose
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig


@register_env("SixCubesTwoBoxesOnTable", max_episode_steps=200)
class SixCubesTwoBoxesOnTable(DefaultEnv):

    cube_half_size = 0.018
    centre_tray_2=[-0.3,0.4,0]
    centre_tray_1=[-0.3,-0.4,0]
    size_tray = 0.28
    thickness_tray = 0.02

    table_grid_rows = 3
    table_grid_cols = 3
    table_grid_step = 0.1

    tray_grid_rows = 3
    tray_grid_cols = 3
    tray_grid_step = 0.1

    color_code = dict(
        green = [12/255, 160/255, 42/255, 1],
        yellow = [160/255, 160/255, 12/255, 1],
        red = [1, 0, 0, 1],
        blue = [0, 0, 1, 1],
        white = [1, 1, 1, 1],
        black = [0, 0, 0, 1]
    )
    SUPPORTED_ROBOTS = ["panda", "fetch"]

    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.cubes: List = []
        self.tray: Dict[str, object] = {}
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    def _load_agent(self, options: Dict, initial_agent_poses = sapien.Pose(p=[0,0,0])):
        return super()._load_agent(options, initial_agent_poses)

    def _get_scene_colors(self, options: Dict) -> List[str]:
        return list(
            options.get("colors", random.sample(list(self.color_code.keys()), k=2))
    )

    def _get_cube_counts(self, colors: List[str]) -> Dict[str, int]:
        counts = {color: 0 for color in colors}

        for color_counts in self.composition.values():
            for color, count in color_counts.items():
                counts[color] += int(count)

        return counts

    @property
    def _default_human_render_camera_configs(self):
        # this is just like _sensor_configs, but for adding cameras used for rendering when you call env.render()
        # when render_mode="rgb_array" or env.render_rgb_array()
        # Another feature here is that if there is a camera called render_camera, this is the default view shown initially when a GUI is opened
        pose = sapien_utils.look_at([0.9, 0.7, 0.6], [0.0, 0.2, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        colors = self._get_scene_colors(options)
        self.scene_colors = list(colors)
        self.composition = options.get("composition")
        if self.composition is None:    
            self.composition = self._random_composition()
        self.tray_centers = {
            colors[0]: self.centre_tray_1,
            colors[1]: self.centre_tray_2,
        }
        self.cubes = []
        self.tray = {}

        cube_counts = self._get_cube_counts(colors)

        for color_name in colors:
            for cube_idx in range(1, cube_counts[color_name] + 1):
                cube = actors.build_cube(
                    self.scene,
                    half_size=self.cube_half_size,
                    color=np.array(self.color_code[color_name]),
                    name=f"{color_name}_cube_{cube_idx}",
                    body_type="dynamic",
                    initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
                )
                setattr(self, cube.name, cube)
                self.cubes.append(cube)

        for color_name in colors:
            tray = self._build_tray(
                name=f"{color_name}_tray",
                color=self.color_code[color_name],
            )
            setattr(self, f"{color_name}_tray", tray)
            self.tray[color_name] = tray

    def _get_cube_color(self, cube_name: str) -> str:
        return cube_name.split("_cube_")[0]

    def _build_tray(self, name, color):
        return actors.build_box(
            scene=self.scene,
            half_sizes=np.array(
                [
                    self.size_tray / 2,
                    self.size_tray / 2,
                    self.thickness_tray / 2,
                ],
                dtype=np.float32,
            ),
            color=np.array(color),
            name=name,
            body_type="kinematic",
            initial_pose=sapien.Pose(
                p=[0, 0, self.thickness_tray / 2],
            ),
        )

    def _set_tray_pose(self, batch_size: int, q: List[float]):
        for color_name, center in self.tray_centers.items():
            p = torch.tensor([center[0], center[1], center[2]]).repeat(batch_size, 1)
            pose = Pose.create_from_pq(p=p, q=q)
            self.tray[color_name].set_pose(pose)

    def _grid_offsets(self, rows: int, cols: int, step: float):
        x0 = -(cols - 1) * step / 2
        y0 = -(rows - 1) * step / 2

        return [
            (x0 + col * step, y0 + row * step)
            for row in range(rows)
            for col in range(cols)
        ]

    def _cells_around(self, center, offsets, z):
        return [
            [center[0] + dx, center[1] + dy, z]
            for dx, dy in offsets
        ]

    def _pop_random_cell(self, cells):
        random_index = torch.randint(0, len(cells), (1,)).item()
        return cells.pop(random_index)

    def _table_cells(self):
        offsets = self._grid_offsets(
            rows=self.table_grid_rows,
            cols=self.table_grid_cols,
            step=self.table_grid_step,
        )
        return self._cells_around([0, 0, 0], offsets, self.cube_half_size)

    def _tray_cells(self, color_name: str):
        offsets = self._grid_offsets(
            rows=self.tray_grid_rows,
            cols=self.tray_grid_cols,
            step=self.tray_grid_step,
        )
        return self._cells_around(
            self.tray_centers[color_name],
            offsets,
            self.cube_half_size,
        )


    def _random_composition(self) -> Dict[str, Dict[str, int]]:
        total_table_cubes = random.randint(3, 6)

        composition = {
            "table": {color: 0 for color in self.scene_colors}
        }

        for color in self.scene_colors:
            composition[f"{color}_tray"] = {}

        for _ in range(total_table_cubes):
            color = random.choice(self.scene_colors)
            composition["table"][color] += 1

        return composition

    def _resolved_assignment(self, options: dict) -> Dict[str, List[object]]:
        composition = self.composition

        cubes_by_color = {
            color: [
                cube for cube in self.cubes
                if self._get_cube_color(cube.name) == color
            ]
            for color in self.scene_colors
        }

        next_cube_index = {
            color: 0
            for color in self.scene_colors
        }

        assignment = {
            "table": [],
        }

        for color in self.scene_colors:
            assignment[f"{color}_tray"] = []

        for location, color_counts in composition.items():
            for color, count in color_counts.items():
                for _ in range(int(count)):
                    cube = cubes_by_color[color][next_cube_index[color]]
                    assignment[location].append(cube)
                    next_cube_index[color] += 1

        return assignment

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            self.table_scene.initialize(env_idx)
            q = [1, 0, 0, 0]
            self._set_tray_pose(b, q)

            available_cells = {
                "table": self._table_cells(),
            }

            for color in self.scene_colors:
                available_cells[f"{color}_tray"] = self._tray_cells(color)

            assignment = self._resolved_assignment(options)

            for location, cubes in assignment.items():
                cells = available_cells[location]

                for cube in cubes:
                    cell = self._pop_random_cell(cells)
                    xyz = torch.tensor(cell).repeat(b, 1)
                    obj_pose = Pose.create_from_pq(p=xyz, q=q)
                    cube.set_pose(obj_pose)

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = {cube.name: cube.pose.raw_pose for cube in self.cubes}
        for color_name, tray in self.tray.items():
            obs[f"{color_name}_tray"] = tray.pose.raw_pose
        reference = self.agent.tcp.pose.raw_pose

        table_pose = torch.zeros_like(reference)
        table_pose[:, 3] = 1.0 
        obs["table"] = table_pose

        obs["agent_tcp"] = self.agent.tcp.pose.raw_pose
        return obs

# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict, List, Union
import random
import numpy as np
import sapien
import torch

from magma_core.base.envs import DefaultEnv

from mani_skill.utils.structs import Pose
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder


@register_env("SixCubesTwoBoxesOnTable", max_episode_steps=200)
class SixCubesTwoBoxesOnTable(DefaultEnv):
    """
    Task Description
    ----------------
    The task is to place 3 green and 3 yellow cubes initially lying on a table in boxes
    of the same color.

    Randomizations
    --------------
    Positions of cube are randomized.
    """

    cube_half_size = 0.02
    goal_thresh = 0.025
    centre_box_2=[-0.4,0.3,0]
    centre_box_1=[-0.4,-0.3,0]
    size_box = 0.25
    thickness_box = 0.01

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
        self.boxes: Dict[str, object] = {}
        self.scene_colors: List[str] = ["green", "yellow"]
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    """
    Reconfiguration Code

    below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general. In CPU simulation,
    for some tasks these may need to be called multiple times if you need to swap out object assets. In GPU simulation these will only ever be called once.
    """

    def _load_agent(self, options: Dict, initial_agent_poses = sapien.Pose(p=[0,0,0])):
        return super()._load_agent(options, initial_agent_poses)

    def _get_scene_colors(self, options: Dict) -> List[str]:
        colors_option = options.get("colors", ["green", "yellow"])
        if not isinstance(colors_option, (list, tuple)):
            raise RuntimeError("The colors option must be a list or tuple of exactly 2 colors")

        colors = list(colors_option)
        if len(colors) != 2:
            raise RuntimeError("The colors option must contain exactly 2 colors")
        if len(set(colors)) != len(colors):
            raise RuntimeError("The colors option must not contain duplicates")

        unknown_colors = [color for color in colors if color not in self.color_code]
        if unknown_colors:
            raise RuntimeError(
                f"Unknown colors {unknown_colors}. Supported colors are {sorted(self.color_code.keys())}"
            )

        return colors

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        colors = self._get_scene_colors(options)
        self.scene_colors = list(colors)
        self.box_centers = {
            colors[0]: self.centre_box_1,
            colors[1]: self.centre_box_2,
        }
        self.cubes = []
        self.boxes = {}

        for color_name, y_pos in zip(colors, [-0.05, 0.05]):
            for cube_idx, x_pos in enumerate([-0.1, 0.0, 0.1], start=1):
                cube = actors.build_cube(
                    self.scene,
                    half_size=self.cube_half_size,
                    color=np.array(self.color_code[color_name]),
                    name=f"{color_name}_cube_{cube_idx}",
                    body_type="dynamic",
                    initial_pose=sapien.Pose(p=[x_pos, y_pos, self.cube_half_size]),
                )
                setattr(self, cube.name, cube)
                self.cubes.append(cube)

        for color_name in colors:
            box = self.create_box(
                size=self.size_box,
                thickness=self.thickness_box,
                name=f"{color_name}_box",
                color=self.color_code[color_name],
            )
            setattr(self, f"{color_name}_box", box)
            self.boxes[color_name] = box

    def _get_cube_color(self, cube_name: str) -> str:
        return cube_name.split("_cube_")[0]

    def _set_boxes_pose(self, batch_size: int, q: List[float]):
        for color_name, center in self.box_centers.items():
            p = torch.tensor([center[0], center[1], center[2]]).repeat(batch_size, 1)
            pose = Pose.create_from_pq(p=p, q=q)
            self.boxes[color_name].set_pose(pose)

    def _get_sorted_cube_position(self, color_name: str, slot_idx: int) -> List[float]:
        slot_offsets = [
            (-0.05, -0.05),
            (0.0, 0.0),
            (0.05, 0.05),
        ]
        offset_x, offset_y = slot_offsets[min(slot_idx, len(slot_offsets) - 1)]
        center = self.box_centers[color_name]
        return [center[0] + offset_x, center[1] + offset_y, self.cube_half_size]


    """
    Episode Initialization Code

    below are all functions involved in episode initialization during environment reset called in the same order. As a user
    you can change these however you want for your desired task. Note that these functions are given a env_idx variable.

    `env_idx` is a torch Tensor representing the indices of the parallel environments that are being initialized/reset. This is used
    to support partial resets where some parallel envs might be reset while others are still running (useful for faster RL and evaluation).
    Generally you only need to really use it to determine batch sizes via len(env_idx). ManiSkill helps handle internally a lot of masking
    you might normally need to do when working with GPU simulation. For specific details check out the push_cube.py code
    """
	
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            cube_names = [cube.name for cube in self.cubes]
            nb_cube_completed = min(options.get("nb_cube_completed", 0), len(cube_names))

            self.table_scene.initialize(env_idx)
            q = [1, 0, 0, 0]
            self._set_boxes_pose(b, q)

            r = 0.15

            available_cells = [(-r,-r),(-r,0),(-r,r),
                (0,-r),(0,0),(0,r),
                (r,-r),(r,0),(r,r)]
            completed = random.sample(cube_names, nb_cube_completed)
            sorted_counts = {color_name: 0 for color_name in self.scene_colors}

            for cube in self.cubes:
                if cube.name in completed:
                    cube_color = self._get_cube_color(cube.name)
                    xyz = torch.tensor(
                        self._get_sorted_cube_position(cube_color, sorted_counts[cube_color])
                    ).repeat(b, 1)
                    sorted_counts[cube_color] += 1
                else:
                    #Get a random availaible cell
                    random_index = torch.randint(0, len(available_cells), (1,)).item()
                    # Get the random item
                    random_cell = available_cells[random_index]
                    available_cells.pop(random_index)

                    xyz = torch.tensor([random_cell[0], random_cell[1], self.cube_half_size]).repeat(b, 1)
                    xyz[..., :2] = xyz[..., :2] + torch.rand((b, 2)) * 0.07 - 0.05
    
                obj_pose = Pose.create_from_pq(p=xyz, q=q)
                cube.set_pose(obj_pose)

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = {cube.name: cube.pose.raw_pose for cube in self.cubes}
        for color_name, box in self.boxes.items():
            obs[f"{color_name}_box_pose"] = box.pose.raw_pose
        obs["agent_tcp"] = self.agent.tcp.pose.raw_pose
        return obs

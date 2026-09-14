# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Tuple, Dict
from transforms3d.euler import euler2quat
import numpy as np
import sapien
import torch, random
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from magma_scenarios.scenarios.bi_robot_sorting.brs_attributes import sorting_objects
from magma_core.simulation.envs import DefaultMultiAgentEnv
from pathlib import Path
from magma_scenarios.envs.visual_assets import OBJECT_VISUALS

GRID_STEP = 0.1
offsets = [
    (-GRID_STEP, -GRID_STEP), (-GRID_STEP, 0), (-GRID_STEP, GRID_STEP),
    (0, -GRID_STEP), (0, 0), (0, GRID_STEP),
    (GRID_STEP, -GRID_STEP), (GRID_STEP, 0), (GRID_STEP, GRID_STEP),
]

YELLOW = [1, 1, 0, 1]
RED = [1, 0, 0, 1]
ORANGE = [1, 0.2, 0, 1]
INTACT_STATE = 1
DAMAGED_STATE = 0
STATIC_STATE = -1

@register_env("BiRobotSorting-v1", max_episode_steps=200)
class BiRobotSorting(DefaultMultiAgentEnv):
    """
    **Task Description:**
    The goal is to sort object in their desired target zone. Each robot have access to one zone + the mutual depose zone.
    """

    SUPPORTED_ROBOTS = [("panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda]]
    cube_half_size = 0.02
    mutual_zone_center = [0,0,0]
    left_zone_center = [0,-1.1,0]
    right_zone_center = [0,1.1,0]
    
    def __init__(self,
        *args, 
        robot_uids=("panda", "panda"), 
        robot_init_qpos_noise=0.0, 
        **kwargs
    ):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)
    
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([2.0, 0.8, 0.75], [0.0, 0.1, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(
            self,
            options: dict,
            initial_agent_poses = None
        ):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -0.6, 0]), sapien.Pose(p=[0, 0.6, 0])]
        )

    def _load_scene(self, options: dict):
        self._load_layout_options(options)
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        self.mutual_zone = self.build_a_zone(name='mutual_zone')
        self.right_zone = self.build_a_zone(collision=True, name='right_zone')
        self.left_zone = self.build_a_zone(collision=True, name='left_zone')

        self.banana = [self._build_object(name,YELLOW) for name in sorting_objects["banana"]]
        self.apple = [self._build_object(name,RED) for name in sorting_objects["apple"]]
        self.orange = [self._build_object(name,ORANGE) for name in sorting_objects["orange"]]
        self.all_sorting_objects = [*self.banana,*self.apple,*self.orange]
        self.sorting_objects_by_name = {
            obj.name: obj for obj in self.all_sorting_objects
        }
    def _load_layout_options(self, options: dict):
        self.obj_left = options.get("left_zone", [])
        self.obj_right = options.get("right_zone", [])
        self.obj_mutual = options.get("mutual_zone", [])
        self.damaged_objs = options.get("damaged_objs", [])

    def _select_object_names(self):
        requested_objects = self.obj_left  + self.obj_right + self.obj_mutual
        if not requested_objects:
            random.shuffle(self.all_sorting_objects)
            n = len(self.all_sorting_objects)
            return self.all_sorting_objects[:int(n/3)], self.all_sorting_objects[int(n/3):int(2*n/3)], self.all_sorting_objects[int(2*n/3):n]

        obj_left = [obj for obj in requested_objects if obj in sorting_objects.values()]
        obj_right = [obj for obj in requested_objects if obj in sorting_objects.values()]
        obj_mutual = [obj for obj in requested_objects if obj in sorting_objects.values()]
        return obj_left, obj_right, obj_mutual

    def _build_object(self, name: str, color):
        obj_type = name.split("_")[0]

        return self.build_box_object(
            name=name,
            half_size=(0.02, 0.02, 0.02),
            color=color,
            visual=OBJECT_VISUALS.get(obj_type),
            visual_assets_dir=(
                Path(__file__).resolve().parents[2]
                / "assets"
                / "visuel"
            ),
        )
        
    def _place_objects(self,objects,storage_cells,batch_size,q):
        for obj in objects:
            if isinstance(obj, str):
                obj = self.sorting_objects_by_name[obj]

            self._place_object(obj, storage_cells, z=0.01, batch_size=batch_size, q=q)

    def _place_object(self, obj, cells, z, batch_size, q):
        cell = self._pop_random_cell(cells)
        xyz = torch.tensor([cell[0], cell[1], z]).repeat(batch_size, 1)
        obj.set_pose(Pose.create_from_pq(p=xyz, q=q))

        if obj not in self.objects:
            self.objects.append(obj)

    def _pop_random_cell(self, cells):
        random_index = torch.randint(0, len(cells), (1,)).item()
        return cells.pop(random_index)

    def _sample_default_layout_if_needed(self):
        if self.obj_left or self.obj_right or self.obj_mutual:
            return

        random.shuffle(self.all_sorting_objects)
        n = len(self.all_sorting_objects)
        self.obj_left = self.all_sorting_objects[:int(n/3)]
        self.obj_right = self.all_sorting_objects[int(n/3):int(2*n/3)] 
        self.obj_mutual = self.all_sorting_objects[int(2*n/3):n]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self._load_layout_options(options)

            b = len(env_idx)
            q = [0, 1, 0, 0]

            self.mutual_zone.set_pose(sapien.Pose(p=self.mutual_zone_center, q = [0, 1, 0, 0]))
            self.left_zone.set_pose(sapien.Pose(p=self.left_zone_center, q = [0, 1, 0, 0]))
            self.right_zone.set_pose(sapien.Pose(p=self.right_zone_center, q = [0, 1, 0, 0]))

            left_zone_cells = self._cells_around(self.left_zone_center, offsets)
            right_zone_cells = self._cells_around(self.right_zone_center, offsets)
            mutual_zone_cells = self._cells_around(self.mutual_zone_center, offsets)

            self.table_scene.initialize(env_idx)
            self.objects = []
            self.object_states = self._default_object_states()
            self._sample_default_layout_if_needed()
            self._place_objects(self.obj_left, left_zone_cells, b, q)
            self._place_objects(self.obj_right, right_zone_cells, b, q)
            self._place_objects(self.obj_mutual, mutual_zone_cells, b, q)

            self.left_agent.robot.set_pose(
                sapien.Pose(p=[0, -0.6, 0], q=euler2quat(0, 0, np.pi))
            )
            self.right_agent.robot.set_pose(
                sapien.Pose(p=[0, 0.6, 0], q=euler2quat(0, 0, -np.pi))
            )
            
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = self.cube_half_size

            self._apply_damaged_states()

    def _default_object_states(self):
        return {
            obj.name: torch.full((self.num_envs,), INTACT_STATE, dtype=torch.int32, device=self.device)
            for obj in [*self.all_sorting_objects]
        }

    def _apply_damaged_states(self):
        for obj in self.objects:
            self.object_states[obj.name][:] = (
                DAMAGED_STATE if obj.name in self.damaged_objs else INTACT_STATE
            )

    def _cells_around(self, center, offsets):
        return [(center[0] + dx, center[1] + dy) for dx, dy in offsets]

    def get_magma_extra_state(self):
        if not hasattr(self, "object_states"):
            return {}
        return {
            name: state.clone()
            for name, state in self.object_states.items()
        }

    def set_magma_extra_state(self, state: dict):
        if not hasattr(self, "object_states"):
            return
        for name, value in state.items():
            if name in self.object_states:
                self.object_states[name] = value.clone().to(device=self.device)

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]
    def _get_obs_extra(self, info: dict):
        obs = {
            "left_arm_tcp": {
                "pose": self.left_agent.tcp.pose.raw_pose,
                "state": self._static_state(),
            }
        }
        static_entries = [
            ("right_arm_tcp",self.right_agent.tcp.pose.raw_pose),
            (self.left_zone.name,self.left_zone.pose.raw_pose),
            (self.right_zone.name,self.right_zone.pose.raw_pose),
            (self.mutual_zone.name,self.mutual_zone.pose.raw_pose),
        ]
        for name, pose in static_entries:
            self._add_static_obs(obs, name, pose)

        for obj in self.objects :
            obs[obj.name] = {
                "pose": obj.pose.raw_pose,
                "state": self.object_states[obj.name]
            }
        return obs

    def _add_static_obs(self, obs: Dict, name: str, pose):
        obs[name] = {
            "pose": pose,
            "state": self._static_state()
        }

    def _static_state(self):
        return torch.full((self.num_envs,), STATIC_STATE, dtype=torch.int32, device=self.device)

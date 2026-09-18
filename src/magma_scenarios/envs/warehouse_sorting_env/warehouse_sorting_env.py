# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# Author: Mathieu Zimmermann
from typing import Dict

import numpy as np
import sapien
import torch

from magma_scenarios.scenarios.warehouse_sorting.att import (
    AREAS,
    AREA_POSITIONS,
    AREA_SIZE,
    OBJECT_POSITIONS,
)

from mani_skill.utils.structs import Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from magma_core.simulation.envs import DefaultEnv

WAREHOUSE_OBJECTS = (
    ("ref_obj_1", np.array([80, 140, 240, 255]) / 255),
    ("ref_obj_2", np.array([220, 180, 60, 255]) / 255),
    ("ref_obj_3", np.array([90, 190, 100, 255]) / 255),
)

# register the environment by a unique ID and specify a max time limit. Now once this file is imported you can do gym.make("CustomEnv-v0")
@register_env("SortingCubesWarehouse-v1", max_episode_steps=200)
class WarehouseSortingEnv(DefaultEnv):
    """
    Task Description
    ----------------
    The task is to manipulate cube on a table. Objects are cube, but to change object type we call them reference here.
    There are 3 cubes and 5 containers.

    Randomizations
    --------------
    References are randomly assigned to three fixed, collision-safe positions.
    """
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=0, **kwargs)

    def _load_agent(self, options: Dict, initial_agent_poses = sapien.Pose(p=[0, 0, 0])):
        return super()._load_agent(options, initial_agent_poses)

    @property
    def _default_human_render_camera_configs(self):
        # this is just like _sensor_configs, but for adding cameras used for rendering when you call env.render()
        # when render_mode="rgb_array" or env.render_rgb_array()
        # Another feature here is that if there is a camera called render_camera, this is the default view shown initially when a GUI is opened
        pose = sapien_utils.look_at([1, 0, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
         # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.industrial_objects = [
            self.build_box_object(
                name=name,
                half_size=(self.cube_half_size,) * 3,
                color=color,
                body_type="dynamic",
            )
            for name, color in WAREHOUSE_OBJECTS
        ]

        self.containers = [
            self.create_box(
                size=AREA_SIZE,
                thickness=0.008,
                height=0.05,
                name=area_name,
                initial_pose=sapien.Pose(p=position),
            )
            for area_name, position in zip(AREAS, AREA_POSITIONS)
        ]

	
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            q = [1,0,0,0]
            for container, position in zip(self.containers, AREA_POSITIONS):
                xyz = torch.tensor(position, dtype=torch.float32).repeat(b, 1)
                container.set_pose(Pose.create_from_pq(p=xyz, q=q))

            fixed_positions = torch.tensor(OBJECT_POSITIONS, dtype=torch.float32)
            position_indices = torch.stack([
                torch.randperm(len(OBJECT_POSITIONS))
                for _ in range(b)
            ])
            for object_index, industrial_object in enumerate(self.industrial_objects):
                xyz = fixed_positions[position_indices[:, object_index]]
                industrial_object.set_pose(Pose.create_from_pq(p=xyz, q=q))

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            agent_tcp = self.agent.tcp.pose.raw_pose
        )
        for container in self.containers:
            obs[container.name] = container.pose.raw_pose
        for obj in self.industrial_objects:
            obs[obj.name] = obj.pose.raw_pose
        return obs


@register_env("SortingCubesWarehouseTV-v1", max_episode_steps=200)
class WarehouseSortingEnvTV(WarehouseSortingEnv):
    """Backward-compatible environment ID."""

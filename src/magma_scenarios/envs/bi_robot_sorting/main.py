# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Tuple

import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from magma_core.base.envs import DefaultMultiAgentEnv


@register_env("BiRobotSorting-v1", max_episode_steps=200)
class BiRobotSorting(DefaultMultiAgentEnv):
    """
    **Task Description:**
    The goal is to sort object in their desired target zone. Each robot have access to one zone + the mutual depose zone.
    """

    SUPPORTED_ROBOTS = [("panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda]]
    cube_half_size = 0.02

    def __init__(self,
        *args, 
        robot_uids=("panda", "panda"), 
        robot_init_qpos_noise=0.0, 
        **kwargs
    ):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)
    
    # @property
    # def _default_human_render_camera_configs(self):
    #     pose = sapien_utils.look_at([1.4, 0.8, 0.75], [0.0, 0.1, 0.1])
    #     return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(
            self,
            options: dict,
            initial_agent_poses = None
        ):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="obj_A",
            initial_pose=sapien.Pose(p=[0, 0, 0.02]),
        )

        self.mutual_zone = self.build_a_zone(name='mutual_zone')
        self.right_zone = self.build_a_zone(collision=True, name='right_zone')
        self.left_zone = self.build_a_zone(collision=True, name='left_zone')

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.left_init_qpos = self.left_agent.robot.get_qpos()
            xyz = torch.zeros((b, 3))
            # xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            # # ensure cube is spawned on the left side of the table
            # xyz[:, 1] = -0.15 - torch.rand((b,)) * 0.1 + 0.05
            xyz[:, 2] = self.cube_half_size
            # qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, q = [1, 0, 0, 0]))

            self.mutual_zone.set_pose(sapien.Pose(p=[0,0,0], q = [0, 1, 0, 0]))
            self.left_zone.set_pose(sapien.Pose(p=[0,-1.2,0], q = [0, 1, 0, 0]))
            self.right_zone.set_pose(sapien.Pose(p=[0,1.2,0], q = [0, 1, 0, 0]))

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]
    
    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
            obj_A=self.cube.pose.raw_pose,
            left_zone=self.left_zone.pose.raw_pose,
            right_zone=self.right_zone.pose.raw_pose,
            mutual_zone=self.mutual_zone.pose.raw_pose,
        )
        return obs
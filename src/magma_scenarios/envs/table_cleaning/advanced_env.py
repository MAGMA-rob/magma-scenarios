# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from math import pi
from typing import Dict, Tuple

from magma_scenarios.envs.laundry.main import WHITE
import numpy as np
import sapien
import torch
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from transforms3d.euler import euler2quat

from magma_core.simulation.envs import DefaultMultiAgentEnv

from .common import CleanTableCommonMixin, BLUE_LIGHT, RED_LIGHT


@register_env("Advanced_Cleaning_Table", max_episode_steps=250)
class AdvancedCleaningTableEnv(CleanTableCommonMixin, DefaultMultiAgentEnv):
    """
    Multi-robot version of the table cleaning environment.

    robot1/table robot:
        clears the table, throws food away, moves dishware to sink/storage,
        wipes the table.

    robot2/dish robot:
        works around the sink, cleans dirty dishware, and puts it in drying zone.
    """

    SUPPORTED_ROBOTS = [("panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda]]

    sink_center = [0.35, -0.35, 0.02]
    drying_zone_center = [-0.45, -0.35]

    
    table_robot_pose = sapien.Pose(
        p=[-0.6, 0.05, 0],
        q=euler2quat(0, 0, pi/2),
    )

    dish_robot_pose = sapien.Pose(
        p=[0, -0.7, 0],
        q=euler2quat(0, 0, -pi/2),
    )

    def __init__(self,*args,robot_uids=("panda", "panda"),robot_init_qpos_noise=0,**kwargs):
        super().__init__(*args,robot_uids=robot_uids,robot_init_qpos_noise=robot_init_qpos_noise,**kwargs)

    def _load_agent(self, options: dict, initial_agent_poses=None):
        super()._load_agent(
            options,
            [
                self.dish_robot_pose,
                self.table_robot_pose,
            ],
        )

    def _load_scene(self, options: dict):
        self._load_common_scene(options)

        self.sink = self._build_storage_surface(
            "sink",
            self.sink_center,
            WHITE,
            h = 0.02
        )
        self.drying_zone = self._build_storage_surface(
            "drying_zone",
            self.drying_zone_center,
            WHITE,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._load_layout_options(options)
        self._initialize_common_episode(env_idx)
        self._initialize_common_fixed_actors()
        self._initialize_robot_poses()
   
    def _initialize_robot_poses(self):
        self.table_agent.robot.set_pose(
            self.table_robot_pose
        )

        self.dish_agent.robot.set_pose(
            self.dish_robot_pose
        )

    @property
    def table_agent(self) -> Panda:
        return self.agent.agents[1]

    @property
    def dish_agent(self) -> Panda:
        return self.agent.agents[0]

    def _get_obs_extra(self, info: Dict):
        obs = {
            "table_arm_tcp": {
                "pose": self.table_agent.tcp.pose.raw_pose,
                "state": self._static_state(),
            },
            "dish_arm_tcp": {
                "pose": self.dish_agent.tcp.pose.raw_pose,
                "state": self._static_state(),
            },
        }

        self._add_common_static_obs(obs)

        self._add_static_obs(
            obs,
            self.sink.name,
            self.sink.pose.raw_pose,
        )
        self._add_static_obs(
            obs,
            self.drying_zone.name,
            self.drying_zone.pose.raw_pose,
        )

        self._add_object_obs(obs)
        return obs

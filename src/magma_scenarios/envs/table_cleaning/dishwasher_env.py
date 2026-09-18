# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from math import pi
from typing import Dict

import sapien
import torch
from mani_skill.utils.registration import register_env
from transforms3d.euler import euler2quat

from magma_core.simulation.envs import DefaultEnv
from magma_scenarios.envs.asset_lib import create_wash_machine

from .common import CleanTableCommonMixin


@register_env("Cleaning_Table", max_episode_steps=200)
class CleaningTableEnv(CleanTableCommonMixin, DefaultEnv):
    washing_machine_pose = [-0.8, -0.5, 0]
    washing_machine_collision_pose = [-0.8, -0.5, 0.02]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0, **kwargs):
        super().__init__(*args,robot_uids=robot_uids,robot_init_qpos_noise=robot_init_qpos_noise,**kwargs)

    def _load_scene(self, options: dict):
        self._load_common_scene(options)

        self.machine_actor = create_wash_machine(self.scene)
        self.wash_machine_collision = self._build_collision_box(
            "washing_machine",
            self.washing_machine_collision_pose,
            0.3,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._load_layout_options(options)
        self._initialize_common_episode(env_idx)
        self._initialize_fixed_actors()

    def _initialize_fixed_actors(self):
        self._initialize_common_fixed_actors()
        self._set_first_joint(self.trash, pi)

        self.machine_actor.set_pose(
            sapien.Pose(
                p=self.washing_machine_pose,
                q=euler2quat(0, 0, -pi / 2),
            )
        )
        self._set_first_joint(self.machine_actor, pi / 2)

    def _get_obs_extra(self, info: Dict):
        obs = {
            "agent_tcp": {
                "pose": self.agent.tcp.pose.raw_pose,
                "state": self._static_state(),
            }
        }

        self._add_common_static_obs(obs)

        self._add_static_obs(
            obs,
            self.wash_machine_collision.name,
            self.wash_machine_collision.pose.raw_pose,
        )
        self._add_object_obs(obs)
        return obs

# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Tuple

import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from magma_core.base.envs import DefaultMultiAgentEnv
from magma_scenarios.scene_builders import MultipleHallSceneBuilder

@register_env("4HallSorting-v1", max_episode_steps=200)
class HallSorting(DefaultMultiAgentEnv):
    """
    **Task Description:**
    The goal is to sort object in their desired target zone. Object can arrive to a zone and need to be sended to another.
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

    def _load_agent(
            self,
            options: dict,
            initial_agent_poses = None
        ):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )

    def _load_scene(self, options: dict):
        self.table_scene = MultipleHallSceneBuilder(env=self)
        self.table_scene.build(4)
        self.crateA = [
            actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name=f"crateA_{i}",
            initial_pose=sapien.Pose(p=[0, 0, i*0.02]),
        ) for i in range(2)]
        self.crateB = [
            actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0, 1, 0, 1],
            name=f"crateB_{i}",
            initial_pose=sapien.Pose(p=[0, 0.1, i*0.02]),
        ) for i in range(2)]
        self.crateC = [
            actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 1, 1],
            name=f"crateC_{i}",
            initial_pose=sapien.Pose(p=[0, -0.1, i*0.02]),
        ) for i in range(2)]


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # pass
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
        q = [1, 0, 0, 0]

        if options.get('grouped',False):
            for crate_list in [self.crateA, self.crateB, self.crateC]:
                list_xyz = self.table_scene.compute_random_pose(len(crate_list))
                for i, xyz in enumerate(list_xyz):
                    p = Pose.create_from_pq(p=xyz, q=q)
                    crate_list[i].set_pose(p)
        else:
            for obj in self.crateA + self.crateB + self.crateC:
                xyz = self.table_scene.compute_random_pose(1)[0]
                p = Pose.create_from_pq(p=xyz, q=q)
                obj.set_pose(p)

    
    def _get_obs_extra(self, info: dict):
        obs = self.table_scene.get_default_obs()
        for obj in self.crateA + self.crateB + self.crateC:
            obs[obj.name] = obj.pose.raw_pose 
        return obs
    

    def move_robot_to_slot(self, agent_idx : int, hall_idx : int, env_id : int) -> str:
        return self.table_scene.move_robot_to_slot(agent_idx, hall_idx,env_id)
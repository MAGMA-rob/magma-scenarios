# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

#Author: Mathieu Zimmermann
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.utils.building import actors
from mani_skill.utils.structs import Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.registration import register_env

from magma_core.base.envs import DefaultEnv

containers_poses = [[-1,-0.25,-0.1], [-1,0.0,-0.1], [-1,0.25,-0.1],[-1,-0.5,-0.1], [-1,0.5,-0.1]]

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
    Positions of references are randomized.
    """
    cube_half_size = 0.02
    size_box = 0.2
    thickness_box = 0.01

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=0, **kwargs)

    def _load_agent(self, options: Dict, initial_agent_poses = sapien.Pose(p=[0, 0, 0])):
        return super()._load_agent(options, initial_agent_poses)

    def _load_scene(self, options: dict):
         # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Create three cubes with random names
        self.industrial_objects = [
            actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array([12, 42, 160, 255]) / 255,
                name="ref_obj_1",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
            ),
            actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array([12, 42, 160, 255]) / 255,
                name="ref_obj_2",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
            ),
            actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array([12, 42, 160, 255]) / 255,
                name="ref_obj_3",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
            )
        ]

        self.containers = []
        for i in range(len(containers_poses)):
            pose = np.array(containers_poses[i])
            self.containers.append(
                self.create_box(size=self.size_box, initial_pose=pose, thickness=self.thickness_box, name=f"area{i + 1}", add_bottom_wall=True)
            )
	
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            r = 0.15
            q = [1, 0, 0, 0]
            for i in range(len(containers_poses)):
                p = containers_poses[i]
                p_batched = torch.tensor(p).repeat(b,1)
                self.containers[i].set_pose(Pose.create_from_pq(p=p_batched,q=q))

            available_cells = [(-r,-r),(-r,0),(-r,r),
                (0,-r),(0,0),(0,r),
                (r,-r),(r,0),(r,r)]
            
            for elem_list in [self.industrial_objects]:
                for elem in elem_list:
                    #Get a random availaible cell
                    random_index = torch.randint(0, len(available_cells), (1,)).item()
                    # Get the random item
                    random_cell = available_cells[random_index]
                    available_cells.pop(random_index)

                    xyz = torch.tensor([random_cell[0], random_cell[1], self.cube_half_size]).repeat(b, 1)
                    xyz[..., :2] = xyz[..., :2] + torch.rand((b, 2)) * 0.05 - 0.05

                    obj_pose = Pose.create_from_pq(p=xyz, q=q)
                    
                    elem.set_pose(obj_pose)


    def _get_obs_extra(self, info: Dict):
        obs = dict(
            agent_tcp = self.agent.tcp.pose.raw_pose
        )
        for container in self.containers:
            obs[container.name] = container.pose.raw_pose
        for obj in self.industrial_objects:
            obs[obj.name] = obj.pose.raw_pose
        return obs
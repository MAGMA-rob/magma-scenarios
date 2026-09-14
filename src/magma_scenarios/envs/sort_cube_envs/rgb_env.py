# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils.building import actors
from mani_skill.utils.structs import Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.registration import register_env

from magma_core.simulation.envs import DefaultEnv


# register the environment by a unique ID and specify a max time limit. Now once this file is imported you can do gym.make("CustomEnv-v0")
@register_env("SortCubesRGB-v1", max_episode_steps=200)
class SortRGBCube(DefaultEnv):
    """
    Task Description
    ----------------
    The task is to manipulate cube on a table.
    There are 1 red cube, 1 blue cube, 1 green cube and 1 white box.

    Randomizations
    --------------
    Positions of cubes are randomized.
    """

    cube_half_size = 0.02
    goal_thresh = 0.025
    centre_box=[-0.4,0.3,0]
    size_box = 0.25
    thickness_box = 0.01

    SUPPORTED_ROBOTS = ["panda", "fetch"]

    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0., **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise = robot_init_qpos_noise, **kwargs)

    def _load_agent(self, options: Dict, initial_agent_poses = sapien.Pose(p=[0, 0, 0])):
        return super()._load_agent(options, initial_agent_poses)

    def _load_scene(self, options: dict):
         # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cubeBleu = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="blue_cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        self.cubeRouge = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array([160, 12, 42, 255]) / 255,
                name="red_cube",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[-0.1, 0, self.cube_half_size]),
            )
        
        self.cubeVert = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([42, 160, 12, 255]) / 255,
            name="green_cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.1, 0, self.cube_half_size]),
        )


        self.box = self.create_box(size=self.size_box, thickness=self.thickness_box, name="white_box")
	
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            self.table_scene.initialize(env_idx)
            p = torch.tensor([self.centre_box[0],self.centre_box[1],self.centre_box[2]]).repeat(b,1)
            q = [1, 0, 0, 0]
            pose = Pose.create_from_pq(p=p,q=q)
            self.box.set_pose(pose)
            
            r = 0.15

            available_cells = [(-r,-r),(-r,0),(-r,r),
                (0,-r),(0,0),(0,r),
                (r,-r),(r,0),(r,r)]

            cubes = ["Vert","Rouge","Bleu"]
            
            for i in range(3):
                #Get a random availaible cell
                random_index = torch.randint(0, len(available_cells), (1,)).item()
                # Get the random item
                random_cell = available_cells[random_index]
                available_cells.pop(random_index)

                xyz = torch.tensor([random_cell[0], random_cell[1], self.cube_half_size]).repeat(b, 1)
                xyz[..., :2] = xyz[..., :2] + torch.rand((b, 2)) * 0.1 - 0.05
    
                obj_pose = Pose.create_from_pq(p=xyz, q=q)
                
                cube = getattr(self, f"cube{cubes[i]}")
                cube.set_pose(obj_pose)

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            blue_cube=self.cubeBleu.pose.raw_pose,
            white_box=self.box.pose.raw_pose,
            green_cube=self.cubeVert.pose.raw_pose,
            red_cube=self.cubeRouge.pose.raw_pose,
            agent_tcp=self.agent.tcp.pose.raw_pose,
        )
        return obs

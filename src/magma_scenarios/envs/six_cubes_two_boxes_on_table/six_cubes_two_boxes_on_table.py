# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Dict, Union
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
    centre_yellow_box=[-0.4,0.3,0]
    centre_green_box=[-0.4,-0.3,0]
    size_box = 0.25
    thickness_box = 0.01

    green = [12/255, 160/255, 42/255, 1]
    yellow = [160/255, 160/255, 12/255, 1]
    SUPPORTED_ROBOTS = ["panda", "fetch"]

    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    """
    Reconfiguration Code

    below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general. In CPU simulation,
    for some tasks these may need to be called multiple times if you need to swap out object assets. In GPU simulation these will only ever be called once.
    """

    def _load_agent(self, options: Dict, initial_agent_poses = sapien.Pose(p=[0,0,0])):
        return super()._load_agent(options, initial_agent_poses)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.green_cube_1 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array(self.green),
            name="green_cube_1",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[-0.1, -0.05, self.cube_half_size]),
        )

        self.green_cube_2 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array(self.green),
            name="green_cube_2",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, -0.05, self.cube_half_size]),
        )

        self.green_cube_3 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array(self.green),
            name="green_cube_3",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.1, -0.05, self.cube_half_size]),
        )
        
        self.yellow_cube_1 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array(self.yellow),
            name="yellow_cube_1",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[-0.1, 0.05, self.cube_half_size]),
        )

        self.yellow_cube_2 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array(self.yellow),
            name="yellow_cube_2",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0.05, self.cube_half_size]),
        )

        self.yellow_cube_3 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array(self.yellow),
            name="yellow_cube_3",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.1, 0.05, self.cube_half_size]),
        )


        self.green_box = self.create_box(size=self.size_box, thickness=self.thickness_box, name="green_box", color = self.green)
        self.yellow_box = self.create_box(size=self.size_box, thickness=self.thickness_box, name="yellow_box", color = self.yellow)


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

            self.table_scene.initialize(env_idx)
            q = [1, 0, 0, 0]
            # Set pose of yellow box
            p = torch.tensor([self.centre_yellow_box[0],
                              self.centre_yellow_box[1],
                              self.centre_yellow_box[2]]).repeat(b,1)
            pose = Pose.create_from_pq(p=p,q=q)
            self.yellow_box.set_pose(pose)
            # Set pose of green box
            p = torch.tensor([self.centre_green_box[0],
                              self.centre_green_box[1],
                              self.centre_green_box[2]]).repeat(b,1)
            pose = Pose.create_from_pq(p=p,q=q)
            self.green_box.set_pose(pose)

            r = 0.15

            available_cells = [(-r,-r),(-r,0),(-r,r),
                (0,-r),(0,0),(0,r),
                (r,-r),(r,0),(r,r)]

            cubes = ["green_cube_1", "green_cube_2", "green_cube_3",
                     "yellow_cube_1", "yellow_cube_2", "yellow_cube_3"]
            
            for i in range(6):
                #Get a random availaible cell
                random_index = torch.randint(0, len(available_cells), (1,)).item()
                # Get the random item
                random_cell = available_cells[random_index]
                available_cells.pop(random_index)

                xyz = torch.tensor([random_cell[0], random_cell[1], self.cube_half_size]).repeat(b, 1)
                xyz[..., :2] = xyz[..., :2] + torch.rand((b, 2)) * 0.07 - 0.05
    
                obj_pose = Pose.create_from_pq(p=xyz, q=q)
                
                cube = getattr(self, cubes[i])
                cube.set_pose(obj_pose)

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            green_cube_1_pose=self.green_cube_1.pose.raw_pose,
            green_cube_2_pose=self.green_cube_2.pose.raw_pose,
            green_cube_3_pose=self.green_cube_3.pose.raw_pose,
            yellow_cube_1_pose=self.yellow_cube_1.pose.raw_pose,
            yellow_cube_2_pose=self.yellow_cube_2.pose.raw_pose,
            yellow_cube_3_pose=self.yellow_cube_3.pose.raw_pose,
            green_box_pose=self.green_box.pose.raw_pose,
            yellow_box_pose=self.yellow_box.pose.raw_pose,
            agent_tcp=self.agent.tcp.pose.raw_pose
        )
        return obs

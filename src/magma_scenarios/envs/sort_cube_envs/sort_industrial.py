# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from magma_core.simulation.envs import DefaultEnv

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.structs import Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

containerA=[-0.4,0.3,0]
containerB=[-0.4,-0.3,0]

# register the environment by a unique ID and specify a max time limit. Now once this file is imported you can do gym.make("CustomEnv-v0")
@register_env("SortCubesIndustrial-v1", max_episode_steps=200)
class CompleteSortCubesEnv(DefaultEnv):
    """
    Task Description
    ----------------
    The task is to manipulate cube on a table. Objects are cube, but to change object type we call them reference here.
    There are 3 reference 2350 and 2 reference 2450 and two container A and B.
    It's an industrial version of classic sortCube.

    Randomizations
    --------------
    Positions of references are randomized.
    """

    cube_half_size = 0.02
    goal_thresh = 0.025
    size_box = 0.25
    thickness_box = 0.01

    # here you can define a list of robots that this task is built to support and be solved by. This is so that
    # users won't be permitted to use robots not predefined here. If SUPPORTED_ROBOTS is not defined then users can do anything
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    # if you want to say you support multiple robots you can use SUPPORTED_ROBOTS = [["panda", "panda"], ["panda", "fetch"]] etc.

    # to help with programming, you can assert what type of agents are supported like below, and any shared properties of self.agent
    # become available to typecheckers and auto-completion. E.g. Panda and Fetch both share a property called .tcp (tool center point).
    agent: Union[Panda, Fetch]
    # if you want to do typing for multi-agent setups, use this below and specify what possible tuples of robots are permitted by typing
    # this will then populate agent.agents (list of the instantiated agents) with the right typing
    # agent: MultiAgent[Union[Tuple[Panda, Panda], Tuple[Panda, Panda, Panda]]]

    # in the __init__ function you can pick a default robot your task should use e.g. the panda robot by setting a default for robot_uids argument
    # note that if robot_uids is a list of robot uids, then we treat it as a multi-agent setup and load each robot separately.
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0., **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=0., **kwargs)


    def _load_agent(self, options: Dict, initial_agent_poses = sapien.Pose(p=[0, 0, 0])):
        return super()._load_agent(options, initial_agent_poses)

    def _load_scene(self, options: dict):
         # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.ref_2350 = [
            actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array([12, 42, 160, 255]) / 255,
                name="ref_2350_1",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
            ),
            actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array([12, 42, 160, 255]) / 255,
                name="ref_2350_2",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
            ),
            actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array([12, 42, 160, 255]) / 255,
                name="ref_2350_3",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
            )
        ]

        self.ref_2450 = [
            actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array([160, 12, 42, 255]) / 255,
                name="ref_2450_1",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[-0.1, 0, self.cube_half_size]),
            ),
            actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array([160, 12, 42, 255]) / 255,
                name="ref_2450_2",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0.1, 0, self.cube_half_size]),
            )
        ]
        

        self.containers = [
            self.create_box(initial_pose=sapien.Pose(p=containerA),size=self.size_box, thickness=self.thickness_box, name="containerA"),
            self.create_box(initial_pose=sapien.Pose(p=containerB),size=self.size_box, thickness=self.thickness_box, name="containerB")
        ]
        
	
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)
            pA = torch.tensor(containerA).repeat(b,1)
            pB = torch.tensor(containerB).repeat(b,1)
            q = [1, 0, 0, 0]

            self.containers[0].set_pose(Pose.create_from_pq(p=pA,q=q))
            self.containers[1].set_pose(Pose.create_from_pq(p=pB,q=q))
            
            r = 0.15

            available_cells = [(-r,-r),(-r,0),(-r,r),
                (0,-r),(0,0),(0,r),
                (r,-r),(r,0),(r,r)]
            
            for elem_list in [self.ref_2350, self.ref_2450]:
                for elem in elem_list:
                    #Get a random availaible cell
                    random_index = torch.randint(0, len(available_cells), (1,)).item()
                    # Get the random item
                    random_cell = available_cells[random_index]
                    available_cells.pop(random_index)

                    # here we write some randomization code that randomizes the x, y position of the cube we are pushing
                    # in the range [-0.1, -0.1] to [0.1, 0.1]
                    xyz = torch.tensor([random_cell[0], random_cell[1], self.cube_half_size]).repeat(b, 1)
                    xyz[..., :2] = xyz[..., :2] + torch.rand((b, 2)) * 0.1 - 0.05
        
                    

                    # we can then create a pose object using Pose.create_from_pq to then set the cube pose with. Note that even though our quaternion
                    # is not batched, Pose.create_from_pq will automatically batch p or q accordingly
                    # furthermore, notice how here we do not even using env_idx as a variable to say set the pose for objects in desired
                    # environments. This is because internally any calls to set data on the GPU buffer (e.g. set_pose, set_linear_velocity etc.)
                    # automatically are masked so that you can only set data on objects in environments that are meant to be initialized
                    obj_pose = Pose.create_from_pq(p=xyz, q=q)
                    
                    elem.set_pose(obj_pose)

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            containerA=self.containers[0].pose.raw_pose,
            containerB=self.containers[1].pose.raw_pose,
        )
        for obj in self.ref_2350:
            obs[obj.name] = obj.pose.raw_pose
        for obj in self.ref_2450:
            obs[obj.name] = obj.pose.raw_pose
        
        return obs
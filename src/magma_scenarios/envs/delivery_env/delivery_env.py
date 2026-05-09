# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from os import name
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from magma_core.base.envs import DefaultEnv

from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils.building import actors
from mani_skill.utils.structs import Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.registration import register_env
from magma_scenarios.envs.asset_lib import create_cardboard_box_builder, create_donut

container_pose = [-1,-0.25,0]

# register the environment by a unique ID and specify a max time limit. Now once this file is imported you can do gym.make("CustomEnv-v0")
@register_env("DeliveryBase-v1", max_episode_steps=200)
class DeliveryEnv(DefaultEnv):
    """
    Env Description
    ----------------
    The env is composed of 8 cubes and 1 box. The idea is to put cubes in the box following a varying recipe.

    Randomizations
    --------------
    Positions of references are randomized.
    """

    cube_half_size = 0.015
    cylinder_half_length = 0.04
    cylinder_radius = 0.015
    size_box = 0.25
    thickness_box = 0.01


    SUPPORTED_ROBOTS = ["panda", "fetch"]
    USE_TV_OBJECTS = False

    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)


    """
    Reconfiguration Code

    below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general. In CPU simulation,
    for some tasks these may need to be called multiple times if you need to swap out object assets. In GPU simulation these will only ever be called once.
    """

    def _load_agent(self, options: Dict, initial_agent_poses = sapien.Pose(p=[0, 0, 0])):
        return super()._load_agent(options, initial_agent_poses)

    def _build_cube_object(self, name: str, color: np.ndarray):
        return actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=color,
            name=name,
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

    def _load_scene(self, options: dict):
         # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        coca_color = np.array([240, 10, 10, 255]) / 255
        ice_tea_color = np.array([240, 240, 75, 255]) / 255
        donut_color = np.array([180, 120, 20, 255]) / 255
        brets_color = np.array([65, 180, 75, 255]) / 255

        if self.USE_TV_OBJECTS:
            self.cocas = [
                actors.build_cylinder(
                    self.scene,
                    radius=self.cylinder_radius,
                    half_length=self.cylinder_half_length,
                    color=coca_color,
                    name="coca_1",
                    body_type="dynamic",
                    initial_pose=sapien.Pose(p=[0, 0, self.cylinder_half_length]),
                ),
                actors.build_cylinder(
                    self.scene,
                    radius=self.cylinder_radius,
                    half_length=self.cylinder_half_length,
                    color=coca_color,
                    name="coca_2",
                    body_type="dynamic",
                    initial_pose=sapien.Pose(p=[0, 0, self.cylinder_half_length]),
                )
            ]

            self.ice_tea = [
                actors.build_cylinder(
                    self.scene,
                    radius=self.cylinder_radius,
                    half_length=self.cylinder_half_length,
                    color=ice_tea_color,
                    name="icetea_1",
                    body_type="dynamic",
                    initial_pose=sapien.Pose(p=[0, 0, self.cylinder_half_length]),
                ),
                actors.build_cylinder(
                    self.scene,
                    radius=self.cylinder_radius,
                    half_length=self.cylinder_half_length,
                    color=ice_tea_color,
                    name="icetea_2",
                    body_type="dynamic",
                    initial_pose=sapien.Pose(p=[0, 0, self.cylinder_half_length]),
                )
            ]

            self.donuts = [
                create_donut(self.scene, 'donut_1', pose=sapien.Pose(p=[0, 0, self.cube_half_size])),
                create_donut(self.scene, 'donut_2', pose=sapien.Pose(p=[0, 0, self.cube_half_size]))
            ]
        else:
            self.cocas = [
                self._build_cube_object("coca_1", coca_color),
                self._build_cube_object("coca_2", coca_color),
            ]
            self.ice_tea = [
                self._build_cube_object("icetea_1", ice_tea_color),
                self._build_cube_object("icetea_2", ice_tea_color),
            ]
            self.donuts = [
                self._build_cube_object("donut_1", donut_color),
                self._build_cube_object("donut_2", donut_color),
            ]

        self.brets = [
            self._build_cube_object("brets_1", brets_color),
            self._build_cube_object("brets_2", brets_color),
        ]
        box_builder = create_cardboard_box_builder(self.scene)
        self.container = box_builder.build(name="container")

        self.objects = []

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
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)
            q = [1, 0, 0, 0]

            p_batched = torch.tensor(container_pose).repeat(b,1)
            self.container.set_pose(Pose.create_from_pq(p=p_batched,q=q))
            
            r = 0.15

            available_cells = [(-r,-r),(-r,0),(-r,r),
                (0,-r),(0,0),(0,r),
                (r,-r),(r,0),(r,r)]
            self.objects = []
            for elem_list in [self.cocas, self.ice_tea, self.donuts, self.brets]:
                self.objects.extend(elem_list)
                for elem in elem_list:
                    #Get a random availaible cell
                    random_index = torch.randint(0, len(available_cells), (1,)).item()
                    # Get the random item
                    random_cell = available_cells[random_index]
                    available_cells.pop(random_index)

                    xyz = torch.tensor([random_cell[0], random_cell[1], self.cube_half_size]).repeat(b, 1)
                    # xyz[..., :2] = xyz[..., :2] + torch.rand((b, 2)) * 0.1 - 0.05
        
                    

                    # we can then create a pose object using Pose.create_from_pq to then set the cube pose with. Note that even though our quaternion
                    # is not batched, Pose.create_from_pq will automatically batch p or q accordingly
                    # furthermore, notice how here we do not even using env_idx as a variable to say set the pose for objects in desired
                    # environments. This is because internally any calls to set data on the GPU buffer (e.g. set_pose, set_linear_velocity etc.)
                    # automatically are masked so that you can only set data on objects in environments that are meant to be initialized
                    obj_pose = Pose.create_from_pq(p=xyz, q=q)
                    
                    elem.set_pose(obj_pose)

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            agent_tcp = self.agent.tcp.pose.raw_pose
        )
        obs[self.container.name] = self.container.pose.raw_pose
        for obj in self.objects:
            obs[obj.name] = obj.pose.raw_pose
        
        return obs


@register_env("DeliveryBaseTV-v1", max_episode_steps=200)
class DeliveryEnvTV(DeliveryEnv):
    USE_TV_OBJECTS = True

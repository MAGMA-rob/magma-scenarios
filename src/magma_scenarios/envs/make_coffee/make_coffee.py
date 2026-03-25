# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# Arthur TANNEAU

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import os

from magma_core.base.envs import DefaultEnv

from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils.structs import Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.building.actors.ycb import get_ycb_builder

from magma_scenarios.envs.asset_lib import create_coffee_maker

from transforms3d.euler import euler2quat

@register_env("MakeCoffee-v1", max_episode_steps=200)
class MakeCoffeeEnv(DefaultEnv):
    """
    Env Description
    ----------------
    The env is composed of a coffee machine, some pods (represented as cubes) and a mug (also represented a cube).

    Randomizations
    --------------
    Nothing is randomized.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # any shared properties of self.agent become available to typecheckers and auto-completion. 
    # E.g. Panda and Fetch both share a property called .tcp (tool center point).
    agent: Union[Panda, Fetch]


    # mug params
    mug_scale = 0.035
    mug_density = 0.2

    # capsules params
    coffee_size = ["long","short","black"]
    capsules_names = ["black", "milky", "white"]

    # pick a default robot your task should use
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    """
    Reconfiguration Code

    below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general. In CPU simulation,
    for some tasks these may need to be called multiple times if you need to swap out object assets. In GPU simulation these will only ever be called once.
    """

    def _load_agent(self, options: Dict, initial_agent_poses = sapien.Pose(p=[-0.615, 0, 0])):
        return super()._load_agent(options, initial_agent_poses)

    def build_an_ycb(self,id,name=None,pose=None):
        builder = get_ycb_builder(
            scene=self.scene, id=id, add_collision=True, add_visual=True
        )
        if not pose:
            pose = sapien.Pose(p=[0.0425623, 0.0050657, 0.03])
        builder.initial_pose = pose
        #root/.maniskill/data/assets/mani_skill2_ycb fichier là pour les mesh d'objet
        if not name:
            name = id
        return builder.build(name=name)
    
    def create_mug(self, name="mug"):
        builder = self.scene.create_actor_builder()
        box_pose = sapien.Pose([0, 0, 0])  
        box_half_size = [self.mug_scale/2, self.mug_scale/2, self.mug_scale/2]
        builder.add_box_collision(pose=box_pose, half_size=box_half_size)
        builder.add_box_visual(pose=box_pose, half_size=box_half_size, material=[0,0,0.5])
        builder.set_initial_pose(box_pose)
        return builder.build_dynamic(name=name)
    
    def create_cube(self, thickness = 0.01, size = 0.02, name="box", material = [0.5,0.5,0.5]):
        builder = self.scene.create_actor_builder()
        box_pose = sapien.Pose([0, 0, 0])  
        box_half_size = [size/2, size/2, size/2]
        builder.add_box_collision(pose=box_pose, half_size=box_half_size)
        builder.add_box_visual(pose=box_pose, half_size=box_half_size, material=material)
        builder.set_initial_pose(box_pose)
        return builder.build_dynamic(name=name)

    def _load_scene(self, options: dict):
         # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # instanciates objects
        self.coffee_maker = create_coffee_maker(self.scene)
        self.mug = self.create_mug()
        self.capsules = []
        
        shade = 8
        for coffee_name in self.capsules_names:
            self.capsules.append(self.create_cube(name = coffee_name, material=[1/shade,0.6/shade,0]))
            shade /= 2

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
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)

            #Monter la position du bras robot
            qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))

            # init coffee maker
            self.coffee_maker.set_pose(Pose.create_from_pq(p=[0.10, 0, 0.14], q=euler2quat(0, 0, 0)))
            self.mug.set_pose(Pose.create_from_pq(p=[-0.25, -0.3, 0.03], q=euler2quat(0, 0, 0)))
            capsule_index = 0
            for capsule in self.capsules:
                capsule.set_pose(Pose.create_from_pq(p=[-0.25, 0.1*capsule_index, 0.03], q=euler2quat(0, 0, 0)))
                capsule_index += 1

        

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """


    def _get_obs_extra(self, info: Dict):
        """ Return the buttons position and joint position as an array indexed on button names."""
        obs = dict(
            agent_tcp = self.agent.tcp.pose.raw_pose
        )
        obs[self.coffee_maker.name] = torch.cat((self.coffee_maker.pose.raw_pose, self.coffee_maker.get_qpos()), dim=1)
        obs[self.mug.name] = self.mug.pose.raw_pose
        for capsule in self.capsules:
            obs[capsule.name] = capsule.pose.raw_pose
        return obs

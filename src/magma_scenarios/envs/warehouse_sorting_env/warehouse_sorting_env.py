# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

#Author: Mathieu Zimmermann
from typing import Any, Dict, Union

from math import pi
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from magma_scenarios.envs.asset_lib import create_cardboard_box_builder, create_jar, create_pen, create_water_bottle

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
    There are 3 cubes and 3 containers.

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
            create_water_bottle(self.scene, "ref_obj_1"),
            create_jar(self.scene, "ref_obj_2"),
            create_pen(self.scene, "ref_obj_3")
        ]

        self.containers = []
        self.cardboard_box = []
        box_builder = create_cardboard_box_builder(scene=self.scene)
        for i in range(len(containers_poses)):
            pose = np.array(containers_poses[i])
            self.containers.append(
                self.create_box(size=self.size_box, initial_pose=pose, thickness=self.thickness_box, name=f"area{i + 1}", add_bottom_wall=True, color=[1,1,1,0])
            )
            self.cardboard_box.append(
                box_builder.build(name=f"box{i + 1}")
            )

	
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)

            r = 0.15
            q = [1,0,0,0]
            for i in range(len(containers_poses)):
                p = containers_poses[i]
                p_batched = torch.tensor(p).repeat(b,1)
                self.containers[i].set_pose(Pose.create_from_pq(p=p_batched,q=q))

            available_cells = [(-r,-r),(-r,0),(-r,r),
                (0,-r),(0,0),(0,r),
                (r,-r),(r,0),(r,r)]
            
            q = euler2quat(0, pi/2, 0)
            for elem_list in [self.industrial_objects]:
                for elem in elem_list:
                    #Get a random availaible cell
                    random_index = torch.randint(0, len(available_cells), (1,)).item()
                    # Get the random item
                    random_cell = available_cells[random_index]
                    available_cells.pop(random_index)

                    # here we write some randomization code that randomizes the x, y position of the cube we are pushing
                    # in the range [-0.1, -0.1] to [0.1, 0.1]
                    xyz = torch.tensor([random_cell[0], random_cell[1], self.cube_half_size]).repeat(b, 1)
                    xyz[..., :2] = xyz[..., :2] + torch.rand((b, 2)) * 0.05 - 0.05

                    # we can then create a pose object using Pose.create_from_pq to then set the cube pose with. Note that even though our quaternion
                    # is not batched, Pose.create_from_pq will automatically batch p or q accordingly
                    # furthermore, notice how here we do not even using env_idx as a variable to say set the pose for objects in desired
                    # environments. This is because internally any calls to set data on the GPU buffer (e.g. set_pose, set_linear_velocity etc.)
                    # automatically are masked so that you can only set data on objects in environments that are meant to be initialized
                    obj_pose = Pose.create_from_pq(p=xyz, q=q)
                    
                    elem.set_pose(obj_pose)

            # set cardboard box position and widely open
            for i in range(len(containers_poses)):
                pose = np.array(containers_poses[i])
                box = self.cardboard_box[i]
                qpos = box.get_qpos()
                qpos[0][0] = 0.6
                qpos[0][1] = 0.6
                qpos[0][2] = 0.6
                qpos[0][3] = 0.6
                box.set_qpos(qpos)
                box.set_pose(Pose.create_from_pq(p=(pose + [0,0,0.06])))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            agent_tcp = self.agent.tcp.pose.raw_pose
        )
        for container in self.containers:
            obs[container.name] = container.pose.raw_pose
        for obj in self.industrial_objects:
            obs[obj.name] = obj.pose.raw_pose
        return obs
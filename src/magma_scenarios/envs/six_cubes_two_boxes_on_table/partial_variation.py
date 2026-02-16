# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Dict, Union

import numpy as np
import sapien, random
import torch

from mani_skill.utils.structs import Pose
from mani_skill.utils.registration import register_env

from .six_cubes_two_boxes_on_table import SixCubesTwoBoxesOnTable


@register_env("PartialSixCubesTwoBoxesOnTable", max_episode_steps=200)
class PartialSixCubesTwoBoxesOnTable(SixCubesTwoBoxesOnTable):
    """
    Task Description
    ----------------
    The task is to place 3 green and 3 yellow cubes initially lying on a table in boxes
    of the same color. Here some cubes are randomly already sorted.

    Randomizations
    --------------
    Positions of cube are randomized.
    """

    size_box = 0.3
    centre_yellow_box=[-0.5,0.3,0]
    centre_green_box=[-0.5,-0.3,0]
	
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        cubes = ["green_cube_1", "green_cube_2", "green_cube_3",
                     "yellow_cube_1", "yellow_cube_2", "yellow_cube_3"]
        
        nb_cube_completed = min(options.get("nb_cube_completed",0), len(cubes)-1)
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)
            q = [1, 0, 0, 0]
            # Set pose of yellow box
            p = torch.tensor([self.centre_yellow_box[0],
                              self.centre_yellow_box[1],
                              self.centre_yellow_box[2]]).repeat(b,1)
            y_pose = Pose.create_from_pq(p=p,q=q)
            self.yellow_box.set_pose(y_pose)
            # Set pose of green box
            p = torch.tensor([self.centre_green_box[0],
                              self.centre_green_box[1],
                              self.centre_green_box[2]]).repeat(b,1)
            g_pose = Pose.create_from_pq(p=p,q=q)
            self.green_box.set_pose(g_pose)

            r = 0.15

            available_cells = [(-r,-r),(-r,0),(-r,r),
                (0,-r),(0,0),(0,r),
                (r,-r),(r,0),(r,r)]
            
            completed = random.sample(cubes, nb_cube_completed)
            
            for i, cube_name in enumerate(cubes):
                if cube_name in completed:
                    if "green" in cube_name:
                        pos = self.centre_green_box
                    else:
                        pos = self.centre_yellow_box
                else:
                    #Get a random availaible cell
                    random_index = torch.randint(0, len(available_cells), (1,)).item()
                    pos = available_cells[random_index]
                    available_cells.pop(random_index)

                # here we write some randomization code that randomizes the x, y position of the cube we are pushing
                # in the range [-0.1, -0.1] to [0.1, 0.1]
                xyz = torch.tensor([pos[0], pos[1], self.cube_half_size]).repeat(b, 1)
    
                obj_pose = Pose.create_from_pq(p=xyz, q=q)  
                cube = getattr(self, cube_name)
                cube.set_pose(obj_pose)

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            green_cube_1=self.green_cube_1.pose.raw_pose,
            green_cube_2=self.green_cube_2.pose.raw_pose,
            green_cube_3=self.green_cube_3.pose.raw_pose,
            yellow_cube_1=self.yellow_cube_1.pose.raw_pose,
            yellow_cube_2=self.yellow_cube_2.pose.raw_pose,
            yellow_cube_3=self.yellow_cube_3.pose.raw_pose,
            green_box_pose=self.green_box.pose.raw_pose,
            yellow_box_pose=self.yellow_box.pose.raw_pose,
            agent_tcp=self.agent.tcp.pose.raw_pose
        )
        return obs


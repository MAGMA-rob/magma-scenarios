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
    centre_box_2=[-0.5,0.3,0]
    centre_box_1=[-0.5,-0.3,0]
	
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        cube_names = [cube.name for cube in self.cubes]
        colors = self._get_scene_colors(options)
        self.box_centers = {
            colors[0]: self.centre_box_1,
            colors[1]: self.centre_box_2,
        }
        nb_cube_completed = min(options.get("nb_cube_completed",0), len(cube_names)-1)
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)
            q = [1, 0, 0, 0]
            self._set_boxes_pose(b, q)


            r = 0.15

            available_cells = [(-r,-r),(-r,0),(-r,r),
                (0,-r),(0,0),(0,r),
                (r,-r),(r,0),(r,r)]
            
            completed = random.sample(cube_names, nb_cube_completed)
            
            for cube in self.cubes:
                if cube.name in completed:
                    pos = self.box_centers[self._get_cube_color(cube.name)]
                else:
                    #Get a random availaible cell
                    random_index = torch.randint(0, len(available_cells), (1,)).item()
                    pos = available_cells[random_index]
                    available_cells.pop(random_index)

                # here we write some randomization code that randomizes the x, y position of the cube we are pushing
                # in the range [-0.1, -0.1] to [0.1, 0.1]
                xyz = torch.tensor([pos[0], pos[1], self.cube_half_size]).repeat(b, 1)
    
                obj_pose = Pose.create_from_pq(p=xyz, q=q)  
                cube.set_pose(obj_pose)

    def _get_obs_extra(self, info: Dict):
        return super()._get_obs_extra(info)

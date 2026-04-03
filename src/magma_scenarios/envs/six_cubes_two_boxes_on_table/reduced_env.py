# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Dict, Union

import numpy as np
import sapien, random
import torch

from mani_skill.utils.structs import Pose
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from .six_cubes_two_boxes_on_table import SixCubesTwoBoxesOnTable


@register_env("MixedColorSorting", max_episode_steps=200)
class PartialSixCubesTwoBoxesOnTable(SixCubesTwoBoxesOnTable):
    """
    Task Description
    ----------------
    The task is to place 2 red and 1 black cubes inside associated box
    of the same color. Here some cubes are randomly bad placed.

    Randomizations
    --------------
    Positions of cube are randomized.
    """

    size_box = 0.4
    centre_black_box=[-0.5,0.3,0]
    centre_red_box=[-0.5,-0.3,0]

    red = [1, 0, 0, 1]
    black = [0,0,0, 1]

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cubes = []
        self.boxes = {}

        cube_specs = [
            ("red_cube_1", self.red, [-0.1, -0.05, self.cube_half_size]),
            ("red_cube_2", self.red, [0.0, -0.05, self.cube_half_size]),
            ("black_cube_1", self.black, [0.1, 0.05, self.cube_half_size]),
        ]
        for cube_name, color, initial_pos in cube_specs:
            cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array(color),
                name=cube_name,
                body_type="dynamic",
                initial_pose=sapien.Pose(p=initial_pos),
            )
            setattr(self, cube_name, cube)
            self.cubes.append(cube)

        self.red_box = self.create_box(size=self.size_box, thickness=self.thickness_box, name="red_box", color=self.red)
        self.black_box = self.create_box(size=self.size_box, thickness=self.thickness_box, name="black_box", color=self.black)
        self.boxes = {
            "red": self.red_box,
            "black": self.black_box,
        }
        

	
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        cube_names = [cube.name for cube in self.cubes]
        
        nb_mixed = min(options.get("nb_mixed",0), len(cube_names))
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)
            q = [1, 0, 0, 0]
            # Set pose of black box
            p = torch.tensor([self.centre_black_box[0],
                              self.centre_black_box[1],
                              self.centre_black_box[2]]).repeat(b,1)
            y_pose = Pose.create_from_pq(p=p,q=q)
            self.black_box.set_pose(y_pose)
            # Set pose of red box
            p = torch.tensor([self.centre_red_box[0],
                              self.centre_red_box[1],
                              self.centre_red_box[2]]).repeat(b,1)
            g_pose = Pose.create_from_pq(p=p,q=q)
            self.red_box.set_pose(g_pose)

            r = 0.15

            available_cells = [
                (0,-r),(0,0),(0,r),
                (r,-r),(r,0),(r,r)]

            
            
            mix = random.sample(cube_names, nb_mixed)
            
            for cube in self.cubes:
                if cube.name in mix:
                    if "black" in cube.name:
                        pos = self.centre_red_box
                    else:
                        pos = self.centre_black_box
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

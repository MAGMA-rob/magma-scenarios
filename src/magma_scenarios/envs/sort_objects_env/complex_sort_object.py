# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from .sort_object_base import SourceSortObjectEnv

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
from mani_skill.utils.building.actors.ycb import get_ycb_builder


# register the environment by a unique ID and specify a max time limit. Now once this file is imported you can do gym.make("CustomEnv-v0")
@register_env("SortObject-v2", max_episode_steps=200)
class SortObjectEnvComplex(SourceSortObjectEnv):
    """
    Task Description
    ----------------
    Obstacles
    """

    obstacles = {
         "obstacle" : {"size":[0.02, 0.02, 0.6],"pos":[-0.25,0.15,0.3],"rot":[1,0,0,0],"color":[1,0,0]},
         "obstacle_mur" : {"size":[0.04, 1, 1],"pos":[-1.05,-0.4,0.5],"rot":[1,0,0,0],"color":[0,0,0]},
    }

    objects = {
            "strawberry" : {"mesh":"012_strawberry","pos":[0, -0.2, 0.04],"rot":[1,0,1,0]},
            "sugar box" : {"mesh":"004_sugar_box","pos":[0, 0, 0.1],"rot":[1,0,0,0]},
            "mustard bottle" : {"mesh":"006_mustard_bottle","pos":[-0.2, -0.1, 0.05],"rot":[1,0,0,0]},
            "clamp" : {"mesh":"050_medium_clamp","pos":[-0.2, 0, 0.05],"rot":[1,0,0,0]},
            "mug" : {"mesh":"025_mug","pos":[0, -0.1, 0.05],"rot":[1,0,0,0]},
            "apple" : {"mesh":"013_apple","pos":[-0.15, -0.2, 0.05],"rot":[1,0,0,0]},
            "banana" : {"mesh":"011_banana","pos":[0.08, 0, 0.015283],"rot":[1,0,0,0]}
       }

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
          super().__init__(*args, robot_uids=robot_uids,robot_init_qpos_noise=robot_init_qpos_noise,**kwargs)

    def create_obstacle(self,half_size = [0.02, 0.02, 0.2], name="obstacle",color=(0,0,0)):
            builder = self.scene.create_actor_builder()

            wall_pose = sapien.Pose([0, 0, 0])

            builder.add_box_collision(pose=wall_pose, half_size=half_size)
            builder.add_box_visual(pose=wall_pose, half_size=half_size,material=color)

            return builder.build_kinematic(name=name)
    
    def _load_scene(self, options: dict):
         # we use a prebuilt scene builder class that automatically loads in a floor and table.
        super()._load_scene(options=options)

        for obj in self.objects:
            pose = sapien.Pose(p=self.objects[obj]['pos'],q=self.objects[obj]['rot'])
            self.objects[obj]['ref'] = self.build_an_ycb(self.objects[obj]['mesh'],obj,pose)

        for obs in self.obstacles:
            half_size = [x/2 for x in self.obstacles[obs]['size']]
            self.obstacles[obs]['ref'] = self.create_obstacle(
                                                half_size=half_size,
                                                name=obs,
                                                color=tuple(self.obstacles[obs]['color']))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx=env_idx,options=options)
        with torch.device(self.device):
            b=len(env_idx)

            q = [1, 0, 0, 0]
            for obs in self.obstacles:
                p = torch.tensor(self.obstacles[obs]['pos']).repeat(b,1)
                pose = Pose.create_from_pq(p=p,q=q)
                self.obstacles[obs]['ref'].set_pose(pose)

            for obj in self.objects:
                p = self.objects[obj]['pos']
                q = self.objects[obj]['rot']

                p = torch.tensor(p).repeat(b,1)
                pose = Pose.create_from_pq(p=p,q=q)

                self.objects[obj]['ref'].set_pose(pose)
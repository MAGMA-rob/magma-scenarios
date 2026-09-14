from pathlib import Path
from typing import Dict

import numpy as np
import sapien
import torch

from magma_core.simulation.envs import DefaultEnv
from magma_scenarios.envs.visual_assets import OBJECT_VISUALS
from magma_scenarios.scenarios.packaging.attributes import (
    DRINKS,
    FRUITS,
    MAIN_COURSE,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose


VISUAL_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "visuel"

FOOD_VISUALS = {
    "banana": "banana",
    "ananas": "orange",
    "apple": "apple",
    "milk": "cup",
    "water": "glass",
    "juice": "drink",
    "chicken": "snack",
    "fish": "snack",
    "pasta": "plate",
}



@register_env("Packaging", max_episode_steps = 200)
class PackagingEnv(DefaultEnv):

    tray_centre = [-0.2, 0.25]
    table_center = [-0.1,-0.2]

    fruits_size = 0.018
    drinks_size = 0.02
    main_course_size = 0.025

    z_half_tray_size = 0.005


    def __init__(self,*args,robot_uids="panda",robot_init_qpos_noise=0,**kwargs) :
        super().__init__(*args,robot_uids=robot_uids,robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    def _load_agent(self,options : Dict, initial_agent_poses = sapien.Pose(p=[0,0,0])):
        return super()._load_agent(options,initial_agent_poses)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env = self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.tray = self.build_box_object(
            name="tray",
            half_size=(0.16, 0.16, self.z_half_tray_size),
            color=np.array([211, 211, 211, 255]) / 255,
            body_type="static",
            initial_xy=tuple(self.tray_centre),
            visual=OBJECT_VISUALS["tray"],
            visual_assets_dir=VISUAL_ASSETS_DIR,
        )
        
        self.fruits = [
            self._build_food_object(
                name,
                self.fruits_size,
                np.array([255, 229, 188, 255]) / 255,
            )
            for name in FRUITS
        ]

        self.drinks = [
            self._build_food_object(
                name,
                self.drinks_size,
                np.array([185, 206, 235, 255]) / 255,
            )
            for name in DRINKS
        ]

        self.main_course = [
            self._build_food_object(
                name,
                self.main_course_size,
                np.array([65, 180, 75, 255]) / 255,
            )
            for name in MAIN_COURSE
        ]

    def _build_food_object(self, name: str, half_size: float, color: np.ndarray):
        return self.build_box_object(
            name=name,
            half_size=(half_size, half_size, half_size),
            color=color,
            visual=OBJECT_VISUALS[FOOD_VISUALS[name]],
            visual_assets_dir=VISUAL_ASSETS_DIR,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)
        q = [1,0,0,0]

        r = 0.11


        offsets = [
            (-r,-r),(-r,0),(-r,r),
            (0,-r),(0,0),(0,r),
            (r,-r),(r,0),(r,r)
        ] 

        

        table_cells = [
            (self.table_center[0] + dx, self.table_center[1] + dy)
            for dx, dy in offsets
        ]



        self.objects = []
        for elem_list in [self.fruits, self.drinks,self.main_course]:
                self.objects.extend(elem_list)
                for elem in elem_list:
                    #Get a random availaible cell
                    random_index = torch.randint(0, len(table_cells), (1,)).item()
                    # Get the random item
                    random_cell = table_cells[random_index]
                    table_cells.pop(random_index)
                    if elem in self.fruits:
                        z = self.fruits_size
                    elif elem in self.drinks:
                        z = self.drinks_size
                    else:
                        z = self.main_course_size
                    xyz = torch.tensor([random_cell[0], random_cell[1], z]).repeat(b, 1)
                    # xyz[..., :2] = xyz[..., :2] + torch.rand((b, 2)) * 0.1 - 0.05
        
                    

                    # we can then create a pose object using Pose.create_from_pq to then set the cube pose with. Note that even though our quaternion
                    # is not batched, Pose.create_from_pq will automatically batch p or q accordingly
                    # furthermore, notice how here we do not even using env_idx as a variable to say set the pose for objects in desired
                    # environments. This is because internally any calls to set data on the GPU buffer (e.g. set_pose, set_linear_velocity etc.)
                    # automatically are masked so that you can only set data on objects in environments that are meant to be initialized
                    obj_pose = Pose.create_from_pq(p=xyz, q=q)
                    
                    elem.set_pose(obj_pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            agent_tcp = self.agent.tcp.pose.raw_pose
        )
        obs[self.tray.name] = self.tray.pose.raw_pose
        for obj in self.objects :
            obs[obj.name] = obj.pose.raw_pose

        
        return obs

# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from collections import defaultdict
from pathlib import Path
import random
from typing import Dict, Union

import numpy as np
import sapien
import torch
from magma_core.simulation.envs import DefaultEnv
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils.structs import Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.registration import register_env
from magma_scenarios.envs.asset_lib import create_cardboard_box_builder
from magma_scenarios.envs.visual_assets import OBJECT_VISUALS

package_pose = [-1,-0.25,0]
PRODUCTS_SLOT_POSITION = [-0.615, 0.0, 0.0]
PACKAGES_SLOT_POSITION = [0, 1, 0.0]

BOX_POSITIONS = [
    [-0.60, 1.5, 0.0],
    [-0.2, 1.5, 0.0],
    [ 0.2, 1.5, 0.0],
    [ 0.60, 1.5, 0.0],
]

PACKAGE_FREE = 0
PACKAGE_HELD = 1

PRODUCT_TYPES = ["coca","icetea","brets","donut"]

PRODUCT_VISUALS = {
    "coca": "drink",
    "icetea": "hygiene",
    "brets": "snack",
    "donut": "orange",
}

VISUAL_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "visuel"

MAX_PER_PRODUCT_TYPE = 3
MAX_OBJECTS_ON_TABLE = 9

GRID_STEP = 0.15

TABLE_OFFSETS = [
    (-GRID_STEP, -GRID_STEP),
    (-GRID_STEP, 0.0),
    (-GRID_STEP, GRID_STEP),
    (0.0, -GRID_STEP),
    (0.0, 0.0),
    (0.0, GRID_STEP),
    (GRID_STEP, -GRID_STEP),
    (GRID_STEP, 0.0),
    (GRID_STEP, GRID_STEP),
]

# register the environment by a unique ID and specify a max time limit. Now once this file is imported you can do gym.make("CustomEnv-v0")
@register_env("DeliveryBase-v1", max_episode_steps=200)
class DeliveryEnv(DefaultEnv):
    cube_half_size = 0.02
    size_box = 0.25
    thickness_box = 0.01


    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)



    def _load_agent(self, options: Dict, initial_agent_poses = sapien.Pose(p=[0, 0, 0])):
        return super()._load_agent(options, initial_agent_poses)

    def _load_scene(self, options: dict):
        
        self._load_layout_options(options)
        self._sample_default_layout_if_needed()
        
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.product_objects = [self._build_object(name)  for name in self.option_table_objects]


        self.packages = []

        for i in range(4):
            box_builder = create_cardboard_box_builder(self.scene)
            package = box_builder.build(name=f"package_{i + 1}")
            self.packages.append(package)

        self.objects = self.product_objects

    def _sample_default_layout_if_needed(self):
     
        if self.option_table_objects:
            return

        available_types = [
            product_type
            for product_type in PRODUCT_TYPES
            for _ in range(MAX_PER_PRODUCT_TYPE)
        ]

        
        selected_types = random.sample(available_types, k=MAX_OBJECTS_ON_TABLE)

        counters = defaultdict(int)
        selected_names = []

        for product_type in selected_types:
            counters[product_type] += 1
            selected_names.append(f"{product_type}_{counters[product_type]}")

        random.shuffle(selected_names)
        self.option_table_objects = selected_names


    def _place_objects(self, env_idx: torch.Tensor):
        batch_size = len(env_idx)
        quaternion = [1, 0, 0, 0]

        available_cells = list(TABLE_OFFSETS)
        random.shuffle(available_cells)

        for product_object, (offset_x, offset_y) in zip(self.product_objects, available_cells):
            position = torch.tensor(
                [
                    offset_x,
                    offset_y,
                    self.cube_half_size,
                ],
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)

            product_object.set_pose(
                Pose.create_from_pq(
                    p=position,
                    q=quaternion,
                )
            )

    def _load_layout_options(self, options: dict):
        self.option_table_objects = list(options.get("table", []))
	

    def _product_type_from_name(self, name: str):
        for product_type in PRODUCT_TYPES:
            if name.startswith(f"{product_type}_"):
                return product_type

        raise ValueError(f"Unknown product name: {name}")

    def _build_object(self, name: str):
        product_type = self._product_type_from_name(name)

        colors = {
            "coca": np.array([240, 10, 10, 255]) / 255,
            "icetea": np.array([240, 240, 75, 255]) / 255,
            "donut": np.array([180, 120, 20, 255]) / 255,
            "brets": np.array([65, 180, 75, 255]) / 255,
        }

        return self.build_box_object(
            name=name,
            half_size=(self.cube_half_size,) * 3,
            color=colors[product_type],
            body_type="dynamic",
            visual=OBJECT_VISUALS[PRODUCT_VISUALS[product_type]],
            visual_assets_dir=VISUAL_ASSETS_DIR,
        )
        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):

            b = len(env_idx)

            self.table_scene.initialize(env_idx)
            q = [1, 0, 0, 0]

            
            for package, position in zip(self.packages, BOX_POSITIONS):
                p_batched = torch.tensor(
                    position,
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(b, 1)

                package.set_pose(
                    Pose.create_from_pq(
                        p=p_batched,
                        q=q,
                    )
                )
            self.package_states = {
                package.name: torch.full(
                    (self.num_envs,),
                    PACKAGE_FREE,
                    dtype=torch.int32,
                    device=self.device,
                )
                for package in self.packages}


            self._place_objects(env_idx)


    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            agent_tcp = self.agent.tcp.pose.raw_pose
        )
        for package in self.packages:
            obs[package.name] = {
                "pose": package.pose.raw_pose,
                "state": self.package_states[package.name],
                "sim_state": package.get_state(),
            }
        for obj in self.objects:
            obs[obj.name] = obj.pose.raw_pose

        batch_size = self.agent.tcp.pose.raw_pose.shape[0]

        products_slot_pose = torch.tensor(
            PRODUCTS_SLOT_POSITION + [1.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        ).repeat(batch_size, 1)

        packages_slot_pose = torch.tensor(
            PACKAGES_SLOT_POSITION + [1.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        ).repeat(batch_size, 1)

        obs["products_slot"] = products_slot_pose
        obs["packages_slot"] = packages_slot_pose

        table_pose = torch.zeros_like(self.agent.tcp.pose.raw_pose)
        table_pose[:, 3] = 1.0  
        obs["table"] = table_pose
        
        return obs

    def get_magma_extra_state(self):
        if not hasattr(self, "package_states"):
            return {}

        return {
            "package_states": {
                name: state.clone()
                for name, state in self.package_states.items()
            }
        }


    def set_magma_extra_state(self, state: dict):
        if (
            "package_states" not in state
            or not hasattr(self, "package_states")
        ):
            return

        for name, value in state["package_states"].items():
            if name in self.package_states:
                self.package_states[name] = value.clone().to(self.device)

@register_env("DeliveryBaseTV-v1", max_episode_steps=200)
class DeliveryEnvTV(DeliveryEnv):
    """Backward-compatible environment ID; visuals are controlled at runtime."""

# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import os
import sapien
import torch
import numpy as np
from math import pi
import mani_skill.envs.scene
from mani_skill.utils.building import actors
from transforms3d.euler import euler2quat
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose as MSPose

from magma_core.base.envs import DefaultEnv
from .observation import ObjectObservation

WHITE = (1.0, 1.0, 1.0, 1.0)
GREY = (0.95, 0.95, 0.95, 1.0)
BEIGE = (245.0 / 255, 245.0 / 255, 220.0 / 255, 1.0)
BLUE_JEAN = (75.0 / 255, 0.0, 130.0 / 255, 1.0)
BLACK = (0, 0, 0, 1)
RED = (1, 0, 0, 1)

@register_env("Laundry-v1")
class LaundryEnv(DefaultEnv):
    """
    [Version simplified with cubes for now]

    In the laundry environment, the IA will have all the tools necessary to wash clothes.

    You will find a washing machine, a bottle of soap and some cloathes on the table.
    You must wash some of them then store the cleaned clothes in a box.
    """

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    # def clothes(self) -> list[str]:
    #     """List of objects to wash."""
    #     return list(map(lambda actor: actor.name, self._clothes))

    def _load_scene(self, options: dict):
        """Create a scene containing
        - a table
        - a detergent bottle
        - a lot of clothes as cubes
        - a box as washing machine"""
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._clothes = [
            actors.build_box(
                self.scene,
                (0.035, 0.035, 0.02),
                WHITE,
                name="shirt",
                initial_pose=sapien.Pose(p=[0, 0, 0.02]),
            ),
            actors.build_box(
                self.scene,
                (0.035, 0.035, 0.02),
                WHITE,
                name="blouse",
                initial_pose=sapien.Pose(p=[0, 0, 0.02]),
            ),
            actors.build_box(
                self.scene,
                (0.035, 0.035, 0.02),
                BLUE_JEAN,
                name="jeans",
                initial_pose=sapien.Pose(p=[0.1, 0, 0.02]),
            ),
            actors.build_box(
                self.scene,
                (0.035, 0.035, 0.02),
                BLUE_JEAN,
                name="pants",
                initial_pose=sapien.Pose(p=[0.1, 0, 0.02]),
            ),
            actors.build_cube(
                self.scene,
                0.03,
                BEIGE,
                name="short",
                initial_pose=sapien.Pose(p=[0.0, 0.2, 0.02]),
            ),
            actors.build_cube(
                self.scene,
                0.03,
                BEIGE,
                name="socks",
                initial_pose=sapien.Pose(p=[0.1, 0.2, 0.02]),
            ),
            actors.build_cube(
                self.scene,
                0.03,
                color=BLACK,
                name="boxer",
                initial_pose=sapien.Pose(p=[0.0, 0.4, 0.02]),
            ),
            actors.build_cube(
                self.scene,
                0.03,
                color=BLACK,
                name="panties",
                initial_pose=sapien.Pose(p=[0.0, 0.4, 0.02]),
            ),
            actors.build_cube(
                self.scene,
                0.03,
                color=RED,
                name="cap",
                initial_pose=sapien.Pose(p=[0.0, 0.4, 0.02]),
            ),
        ]
        self.detergent = actors.build_box(
            self.scene,
            (0.06, 0.03, 0.03),
            WHITE,
            name="OMO",
            initial_pose=sapien.Pose(p=[0, -0.3, 0.02]),
        )

        self.machine_actor = self.create_washmachine()
        self.wash_machine_collision = self.create_box(
            thickness=0.01,
            size=0.3,
            height=0.1,
            name="washing_machine_basket",
            initial_pose=np.array((-0.8, -0.5, 0.02)),
            add_bottom_wall=True
        )
    
    def create_washmachine(self, name="washing_machine"):
        """ Create a washing machine from a SAPIEN urdf file."""

        dir_path = os.path.dirname(os.path.realpath(__file__))
        urdf_path = os.path.join(dir_path, "washmachine-103781/mobility.urdf")

        loader = self.scene.create_urdf_loader()
        loader.scale = 0.5
        loader.fix_root_link = True
        loader.set_material(0.3, 0, 0)
        loader.set_density(1)

        # the .parse function can also parse multiple articulations
        # actors and cameras but we only use the articulations
        articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
        builder = articulation_builders[0]
        builder.initial_pose = sapien.Pose(p=[-0.8, -0.5, 0], q=euler2quat(0, 0, -pi/2))
        
        return builder.build(name=name)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)
        dev = env_idx.device

        self.table_scene.initialize(env_idx)

        for i, clothe in enumerate(self._clothes):
            x = i % 3
            y = int(i / 3)
            xyz = torch.tensor((x * -0.15, y * 0.15, 0), device=dev)
            clothe.set_pose(
                MSPose.create_from_pq(xyz, torch.tensor((1, 0, 0, 0), device=dev))
            )
        
        qpos = self.machine_actor.get_qpos()
        qpos[0] = pi/2 #set the wash machine door (first link) open
        self.machine_actor.set_qpos(qpos)

    def _get_obs_extra(self, info: dict) -> dict[str, ObjectObservation]:
        """The observations contains position of all objects in the scene."""
        obs = {
            "detergent": ObjectObservation(pose=self.detergent.pose.raw_pose),
            "washing_machine": ObjectObservation(pose=self.machine_actor.pose.raw_pose),
            "washing_machine_basket": ObjectObservation(pose=self.wash_machine_collision.pose.raw_pose),
            "agent_tcp": ObjectObservation(pose=self.agent.tcp.pose.raw_pose),
        }
        for clothe in self._clothes:
            obs[clothe.name] = ObjectObservation(pose=clothe.pose.raw_pose)
        return obs

# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# Arthur TANNEAU

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.structs import Pose, Actor
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.building.actors.ycb import get_ycb_builder

from transforms3d.euler import euler2quat
from magma_core.base.envs import DefaultEnv

@register_env("PressButtonBasic-v1", max_episode_steps=200)
class PressButtonsEnv(DefaultEnv):
    """
    Task Description
    ----------------
    The task is to press a button given his id.

    Randomizations
    --------------
    Nothing is randomized.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]

    agent: Union[Panda, Fetch]

    # button params
    nb_button = 5
    button_stroke = 0.011 # course du bouton
    button_friction=0.8 # frottements
    button_damping=0.5 # amortissement
    button_density=0.00001 # densité de la partie mobile du bouton

    # in the __init__ function you can pick a default robot your task should use e.g. the panda robot by setting a default for robot_uids argument
    # note that if robot_uids is a list of robot uids, then we treat it as a multi-agent setup and load each robot separately.
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0., **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=0., **kwargs)


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

    def create_button(self, name="button", material=[0.6, 0.4, 0.8]):
        """ Create an articulated button with the given name and material."""

        size = np.array([0.01, 0.04, 0.05])
        builder : sapien.ArticulationBuilder = self.scene.create_articulation_builder()
        builder.set_initial_pose(sapien.Pose(p=[0, 0, size[2]/2]))

        # button base
        base : sapien.LinkBuilder = builder.create_link_builder()
        base.set_name(name + "_base")
        base.add_box_collision(half_size=size/2)
        base.add_box_visual(half_size=size/2)

        # button mobile part
        btn_builder: sapien.LinkBuilder = builder.create_link_builder(base)
        btn_builder.set_name(name + "_button")
        size = np.array([0.01, 0.03, 0.04])
        btn_builder.add_box_collision(half_size=size/2, density=self.button_density)
        btn_builder.add_box_visual(half_size=size/2, material=material)

        # prismatic joint (who allow only x translation)
        btn_builder.set_joint_name(name + "_joint")
        btn_builder.set_joint_properties(
        'prismatic',
        limits=[[-self.button_stroke,0]],  # joint limits (for each DoF)
        # pose_in_parent refers to the relative transformation from the parent frame to the joint frame
        pose_in_parent=sapien.Pose(p=[0, 0, 0],q=[1,0,0,0]),
        # pose_in_child refers to the relative transformation from the child frame to the joint frame
        pose_in_child=sapien.Pose(p=[-self.button_stroke, 0, 0],q=[1,0,0,0]),
        friction=self.button_friction,
        damping=self.button_damping
        )
        
        return builder.build(name=name)

    def _load_scene(self, options: dict):
         # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # instanciates buttons
        self.buttons = []
        for i in range(self.nb_button):
            btn = self.create_button(name="sw" + str(i), material=[i/self.nb_button,0.2*(self.nb_button-i)/self.nb_button,0])
            self.buttons.append(btn)

	
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
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

            # place buttons on line and with upward orientation
            i = 1
            for button in self.buttons:
                pose = Pose.create_from_pq(p=[0, 0.05*len(self.buttons)-0.1*i, 0.051], q=euler2quat(0, -np.deg2rad(90), 0))
                button.set_pose(pose)
                i = i+1

        
    def _get_obs_extra(self, info: Dict):
        """ Return the buttons position and joint position as an array indexed on button names."""
        obs = dict(
            agent_tcp = self.agent.tcp.pose.raw_pose
        )
        for obj in self.buttons:
            obs[obj.name] = torch.cat((obj.pose.raw_pose, obj.get_qpos()), dim=1)
        return obs
# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# This is a parent env for some child env.

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

class SourceSortObjectEnv(BaseEnv):

    centre_box=[-0.4,0.3,0]
    size_box = 0.25
    thickness_box = 0.01

    # here you can define a list of robots that this task is built to support and be solved by. This is so that
    # users won't be permitted to use robots not predefined here. If SUPPORTED_ROBOTS is not defined then users can do anything
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    # if you want to say you support multiple robots you can use SUPPORTED_ROBOTS = [["panda", "panda"], ["panda", "fetch"]] etc.

    # to help with programming, you can assert what type of agents are supported like below, and any shared properties of self.agent
    # become available to typecheckers and auto-completion. E.g. Panda and Fetch both share a property called .tcp (tool center point).
    agent: Union[Panda, Fetch]
    # if you want to do typing for multi-agent setups, use this below and specify what possible tuples of robots are permitted by typing
    # this will then populate agent.agents (list of the instantiated agents) with the right typing
    # agent: MultiAgent[Union[Tuple[Panda, Panda], Tuple[Panda, Panda, Panda]]]

    # in the __init__ function you can pick a default robot your task should use e.g. the panda robot by setting a default for robot_uids argument
    # note that if robot_uids is a list of robot uids, then we treat it as a multi-agent setup and load each robot separately.
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations. Note that tasks need to tune their GPU memory configurations accordingly
    # in order to save memory while also running with no errors. In general you can start with low values and increase them
    # depending on the messages that show up when you try to run more environments in parallel. Since this is a python property
    # you can also check self.num_envs to dynamically set configurations as well
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    """
    Reconfiguration Code

    below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general. In CPU simulation,
    for some tasks these may need to be called multiple times if you need to swap out object assets. In GPU simulation these will only ever be called once.
    """

    # TODO : Impact du centre sur l'axe z
    def create_box(self, thickness = 0.01, size = 0.2, name="box"):
        builder = self.scene.create_actor_builder()

        half_size = size/2
        wall_heigt = half_size/2
        half_thickness = thickness/2

        wall_pose = sapien.Pose([0, -half_size, wall_heigt/2])  
        wall_half_size = [half_size, half_thickness, wall_heigt]

        builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half_size)

        wall_pose = sapien.Pose([0, half_size, wall_heigt/2])  
        wall_half_size = [half_size, half_thickness, wall_heigt]

        builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half_size)

        wall_pose = sapien.Pose([-half_size, 0, wall_heigt/2])  
        wall_half_size = [half_thickness, half_size + half_thickness, wall_heigt]

        builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half_size)

        wall_pose = sapien.Pose([half_size, 0, wall_heigt/2])
        wall_half_size = [half_thickness, half_size + half_thickness, wall_heigt]

        builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half_size)

        return builder.build_kinematic(name=name)

    def _load_agent(self, options: dict):
        # this code loads the agent into the current scene. You should use it to specify the initial pose(s) of the agent(s)
        # such that they don't collide with other objects initially
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

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

    def _load_scene(self, options: dict):
         # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()      

        self.box = self.create_box(size=self.size_box, thickness=self.thickness_box, name="white box")
        

    @property
    def _default_sensor_configs(self):
        # the camera used by the policy
        pose = sapien_utils.look_at([-0.5, -0.5, 1], [0.0, 0.0, 0])
        pose2= sapien_utils.look_at([-0.5, 0.5, 1], [0, 0, 0])
        return [
             CameraConfig(
            "RGBD_camera_1", pose=pose, width=1280, height=720, fov=125, near=0.01, far=3
                ),
                CameraConfig(
            "RGBD_camera_2", pose=pose2, width=1280, height=720, fov=125, near=0.01, far=3
                )]
            
    @property
    def _default_human_render_camera_configs(self):
        # this is just like _sensor_configs, but for adding cameras used for rendering when you call env.render()
        # when render_mode="rgb_array" or env.render_rgb_array()
        # Another feature here is that if there is a camera called render_camera, this is the default view shown initially when a GUI is opened
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _setup_sensors(self, options: dict):
        # default code here will setup all sensors. You can add additional code to change the sensors e.g.
        # if you want to randomize camera positions
        return super()._setup_sensors(options)

    def _load_lighting(self, options: dict):
        # default code here will setup all lighting. You can add additional code to change the lighting e.g.
        # if you want to randomize lighting in the scene
        return super()._load_lighting(options)

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
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)
            p = torch.tensor([self.centre_box[0],self.centre_box[1],self.centre_box[2]]).repeat(b,1)
            q = [1, 0, 0, 0]
            pose = Pose.create_from_pq(p=p,q=q)
            self.box.set_pose(pose)

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
        

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return {}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # this should be equal to compute_dense_reward / max possible reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    def get_state_dict(self):
        # this function is important in order to allow accurate replaying of trajectories. Make sure to specify any
        # non simulation state related data such as a random 3D goal position you generated
        # alternatively you can skip this part if the environment's rewards, observations, eval etc. are dependent on simulation data only
        # e.g. self.your_custom_actor.pose.p will always give you your actor's 3D position
        state = super().get_state_dict()
        # state["goal_pos"] = add_your_non_sim_state_data_here
        return state

    def set_state_dict(self, state):
        super().set_state_dict(state)

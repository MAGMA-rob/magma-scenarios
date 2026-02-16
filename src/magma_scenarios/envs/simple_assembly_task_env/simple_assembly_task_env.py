# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# Author : Justin GANIVET

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
from mani_skill.utils.structs import Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("SimpleAssemblyTask", max_episode_steps=200)
class SimpleAssemblyTaskEnv(BaseEnv):
    """
    Task Description
    ----------------
    The task is to assemble pieces.
    There are 6 part: A to F.

    Randomizations
    --------------
    Parts' position.
    """

    cube_half_size = 0.02
    goal_thresh = 0.025

    SUPPORTED_ROBOTS = ["panda", "fetch"]

    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

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

    def get_pose(self, pose, delta_pose):
        return sapien.Pose([pose_ + delta_ for pose_, delta_ in zip(pose, delta_pose)])

    def create_box(self, origin=(0, 0, 0), thickness=0.01, size=0.2, name="box"):
        builder = self.scene.create_actor_builder()

        half_size = size/2
        wall_heigt = half_size/2
        half_thickness = thickness/2

        wall_pose = self.get_pose([0, -half_size, wall_heigt/2], origin)
        wall_half_size = [half_size, half_thickness, wall_heigt]

        builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half_size)

        wall_pose = self.get_pose([0, half_size, wall_heigt/2], origin)
        wall_half_size = [half_size, half_thickness, wall_heigt]

        builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half_size)

        wall_pose = self.get_pose([-half_size, 0, wall_heigt/2], origin)
        wall_half_size = [half_thickness, half_size + half_thickness, wall_heigt]

        builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half_size)

        wall_pose = self.get_pose([half_size, 0, wall_heigt/2], origin)
        wall_half_size = [half_thickness, half_size + half_thickness, wall_heigt]

        builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
        builder.add_box_visual(pose=wall_pose, half_size=wall_half_size)

        return builder.build_kinematic(name=name)
    
    def create_cube(self, name, color, size=0.02):
        cube = actors.build_cube(
                self.scene,
                half_size=size,
                color=color,
                name="part_" + name,
                body_type="dynamic",
                initial_pose=sapien.Pose(p=(0, 0, 0))
            )
        self.cubes.append(cube)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cubes = []

        cube_size = 0.02

        self.create_cube(name="A", size=cube_size, color=np.array([255, 0, 0, 255]) / 255)
        self.create_cube(name="B", size=cube_size, color=np.array([0, 255, 0, 255]) / 255)
        self.create_cube(name="C", size=cube_size, color=np.array([0, 0, 255, 255]) / 255)
        self.create_cube(name="D", size=cube_size, color=np.array([255, 0, 255, 255]) / 255)
        self.create_cube(name="E", size=cube_size, color=np.array([255, 255, 0, 255]) / 255)
        self.create_cube(name="F", size=cube_size, color=np.array([0, 255, 255, 255]) / 255)
        
        self.container_pose = (-0.22, 0.4, 0)
        self.container = self.create_box(origin=(0, 0, 0), size=0.2, name=f"container")

        self.objects = []
        self.objects.extend(self.cubes)
        self.objects.append(self.container)

    @property
    def _default_sensor_configs(self):
        return [
        ]

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
            q = [1, 0, 0, 0]

            p_batched = torch.tensor(self.container_pose).repeat(b,1)
            self.container.set_pose(Pose.create_from_pq(p=p_batched,q=q))

            r = 0.15

            available_cells = [(-r,-r),(-r,0),(-r,r),
            (0,-r),(0,0),(0,r),
            (r,-r),(r,0),(r,r)]

            for part in self.cubes:
                #Get a random availaible cell
                random_index = torch.randint(0, len(available_cells), (1,)).item()
                # Get the random item
                random_cell = available_cells[random_index]
                available_cells.pop(random_index)

                # here we write some randomization code that randomizes the x, y position of the cube we are pushing
                # in the range [-0.1, -0.1] to [0.1, 0.1]
                xyz = torch.tensor([random_cell[0], random_cell[1], self.cube_half_size]).repeat(b, 1)
                xyz[..., :2] = xyz[..., :2] + torch.rand((b, 2)) * 0.1 - 0.05

                # we can then create a pose object using Pose.create_from_pq to then set the cube pose with. Note that even though our quaternion
                # is not batched, Pose.create_from_pq will automatically batch p or q accordingly
                # furthermore, notice how here we do not even using env_idx as a variable to say set the pose for objects in desired
                # environments. This is because internally any calls to set data on the GPU buffer (e.g. set_pose, set_linear_velocity etc.)
                # automatically are masked so that you can only set data on objects in environments that are meant to be initialized
                obj_pose = Pose.create_from_pq(p=xyz, q=q)
                part.set_pose(obj_pose)

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """

    def evaluate(self):
        # this function is used primarily to determine success and failure of a task, both of which are optional. If a dictionary is returned
        # containing "success": bool array indicating if the env is in success state or not, that is used as the terminated variable returned by
        # self.step. Likewise if it contains "fail": bool array indicating the opposite (failure state or not) the same occurs. If both are given
        # then a logical OR is taken so terminated = success | fail. If neither are given, terminated is always all False.
        #
        # You may also include additional keys which will populate the info object returned by self.step and that will be given to
        # `_get_obs_extra` and `_compute_dense_reward`. Note that as everything is batched, you must return a batched array of
        # `self.num_envs` booleans (or 0/1 values) for success an dfail as done in the example below

        return {
            "success": torch.tensor([False] * self.num_envs),
            "is_obj_placed": [False] * self.num_envs,
            "is_robot_static": [False] * self.num_envs,
            "is_grasped": [False] * self.num_envs,
        }

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            agent_tcp=self.agent.tcp.pose.raw_pose
        )
        for obj in self.objects:
            obs[obj.name] = obj.pose.raw_pose

        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # this should be equal to compute_dense_reward / max possible reward
        return 0

    def get_state_dict(self):
        # this function is important in order to allow accurate replaying of trajectories. Make sure to specify any
        # non simulation state related data such as a random 3D goal position you generated
        # alternatively you can skip this part if the environment's rewards, observations, eval etc. are dependent on simulation data only
        # e.g. self.your_custom_actor.pose.p will always give you your actor's 3D position
        state = super().get_state_dict()
        # state["goal_pos"] = add_your_non_sim_state_data_here
        return state

    def set_state_dict(self, state):
        # this function complements get_state and sets any non simulation state related data correctly so the environment behaves
        # the exact same in terms of output rewards, observations, success etc. should you reset state to a given state and take the same actions
        # self.goal_pos = state["goal_pos"]
        super().set_state_dict(state)

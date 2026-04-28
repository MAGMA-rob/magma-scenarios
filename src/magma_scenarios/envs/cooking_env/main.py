from typing import Dict, Union
import sapien
import numpy as np
import torch
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from magma_core.base.envs import DefaultEnv
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.building import actors
from mani_skill.utils.structs import Pose

tray_pose = [-0.02, 0.2,0]

@register_env("Cooking", max_episode_steps = 200)
class CookingEnv(DefaultEnv):


    fruits_size = 0.02
    drinks_size = 0.03
    main_course_size = 0.035

    z_half_tray_size = 0.005


    agent: Union[Panda, Fetch]

    def __init__(self,*args,robot_uids="panda",robot_init_qpos_noise=0,**kwargs) :
        super().__init__(*args,robot_uids=robot_uids,robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    def _load_agent(self,options : Dict, initial_agent_poses = sapien.Pose(p=[0,0,0])):
        return super()._load_agent(options,initial_agent_poses)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env = self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.tray = actors.build_box(
                scene=self.scene,
                half_sizes=np.array([0.25, 0.2, self.z_half_tray_size], dtype=np.float32),
                color=np.array([211, 211, 211, 255]) / 255,
                name="tray",
                body_type="static",
                initial_pose=sapien.Pose(p=[0, 0.2, self.z_half_tray_size]),
            )
        
        self.fruits = [
            actors.build_cube(
                self.scene,
                half_size=self.fruits_size,
                color=np.array([255, 229, 188, 255]) / 255,
                name="banana",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[-0.04, -0.1, self.fruits_size]),
            ),
            actors.build_cube(
                self.scene,
                half_size=self.fruits_size,
                color=np.array([255, 229, 188, 255]) / 255,
                name="ananas",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0.02, -0.1, self.fruits_size]),
            ),
            actors.build_cube(
                self.scene,
                half_size=self.fruits_size,
                color=np.array([255, 229, 188, 255]) / 255,
                name="apple",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0.08, -0.1, self.fruits_size]),
            )
        ]

        self.drinks = [
            actors.build_cube(
                self.scene,
                half_size=self.drinks_size,
                color=np.array([185, 206, 235, 255]) / 255,
                name="milk",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[-0.04, -0.2, self.drinks_size]),
            ),
            actors.build_cube(
                self.scene,
                half_size=self.drinks_size,
                color=np.array([185, 206, 235, 255]) / 255,
                name="watter",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0.02, -0.2, self.drinks_size]),
            ),
            actors.build_cube(
                self.scene,
                half_size=self.drinks_size,
                color=np.array([185, 206, 235, 255]) / 255,
                name="juice",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0.08, -0.2, self.drinks_size]),
            )
        ]

        self.main_course = [
            actors.build_cube(
                self.scene,
                half_size=self.main_course_size,
                color=np.array([65, 180, 75, 255]) / 255,
                name="chiken",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[-0.04, -0.3, self.main_course_size]),
            ),
            actors.build_cube(
                self.scene,
                half_size=self.main_course_size,
                color=np.array([65, 180, 75, 255]) / 255,
                name="fish",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0.02, -0.3, self.main_course_size]),
            ),
            actors.build_cube(
                self.scene,
                half_size=self.main_course_size,
                color=np.array([65, 180, 75, 255]) / 255,
                name="pasta",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0.08, -0.3, self.main_course_size]),
            )
        ]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)
        q = [1,0,0,0]

        r = 0.17


        offsets = [
            (-r,-r),(-r,0),(-r,r),
            (0,-r),(0,0),(0,r),
            (r,-r),(r,0),(r,r)
        ] 

        table_center = [0,-0.3]

        table_cells = [
            (table_center[0] + dx, table_center[1] + dy)
            for dx, dy in offsets
        ]

        tray_center = [0.2,0.2]

        tray_cells = [
            (tray_center[0] + dx, tray_center[1] + dy)
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

        p_batched = torch.tensor(tray_pose).repeat(b,1)
        self.tray.set_pose(Pose.create_from_pq(p=p_batched,q=q))

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            agent_tcp = self.agent.tcp.pose.raw_pose
        )
        obs[self.tray.name] = self.tray.pose.raw_pose
        for obj in self.objects :
            obs[obj.name] = obj.pose.raw_pose

        
        return obs
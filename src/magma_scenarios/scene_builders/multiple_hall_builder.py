import os.path as osp
from pathlib import Path
import random

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.structs import Pose

table_height = 0.52
table_lenght = 1.38
table_width = 0.69

class MultipleHallSceneBuilder(SceneBuilder):
    """
    Build a scene for a maximum of 4 different hall per scene.
    A hall is basically two small table. One for depose and the other for spawn.

    Official support is for Panda
    """

    robot_slot : list
    current_pos_per_robot : list # Index per agent id correspond to the id of the robot_slot
    _default_obs = {}
    
    available_slot = []
    original_spawn_slot = []

    panda_qpos = np.array(
            [
                0.0,
                np.pi / 8,
                0,
                -np.pi * 5 / 8,
                0,
                np.pi * 3 / 4,
                np.pi / 4,
                0.04,
                0.04,
            ]
        )

    def build(self, n : int):
        if n>4:
            raise TypeError("Trying to build a scene with more than 4 hall. Not currently supported.")
        
        model_dir = Path(osp.dirname(__file__)) / "assets"
        table_model_file = str(model_dir / "table.glb")

        radius = 2.0  # distance from env center
        self.halls = []
        self.robot_slot = []
        self.current_pos_per_robot = []
        self.original_spawn_slot = []
        for i in range(n):
            builder = self.scene.create_actor_builder()

            table1_pose = sapien.Pose(p=[0,table_width/1.5,0], q=euler2quat(0, 0, np.pi / 2))
            table2_pose = sapien.Pose(p=[0,-table_width/1.5,0], q=euler2quat(0, 0, np.pi / 2))

            builder.add_box_collision(
                pose=sapien.Pose(p=[0, 0, table_height / 2]),
                half_size=(table_lenght / 2, 1.5 * table_width, table_height / 2),
            )

            builder.add_visual_from_file(
                filename=table_model_file,
                scale=[1, 1, 1],
                pose=table1_pose,
            )

            builder.add_visual_from_file(
                filename=table_model_file,
                scale=[1, 1, 1],
                pose=table2_pose,
            )

            theta = 2 * np.pi * i / n
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)

            # Rotate table to face the center
            # yaw = theta + np.pi

            pose = sapien.Pose(
                p=[x, y, -table_height],
                q=euler2quat(0, 0, 0),
            )
            builder.initial_pose = pose
            self.robot_slot.append(sapien.Pose(p=[x,y,0],q=(euler2quat(0, 0, 0))))
            self._default_obs[f"Hall{n}"] = torch.tensor([x, y, table_height, 1, 0, 0, 0])
            displacement = [-0.1, -0.3, 0.3, 0.1]
            hall_slot = []
            for j in range(4):
                hall_slot.append([x+displacement[j], y+table_width/2, table_height])
            self.original_spawn_slot.append(hall_slot)

            table = builder.build_kinematic(name=f"table-workspace-{i}")

            self.halls.append(table)

        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-table_height
        )

        self.scene_objects: list[sapien.Entity] = [*self.halls, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        robot_slot = self.robot_slot.copy()

        panda_qpos = np.array(
            [
                0.0,
                np.pi / 8,
                0,
                -np.pi * 5 / 8,
                0,
                np.pi * 3 / 4,
                np.pi / 4,
                0.04,
                0.04,
            ]
        )

        
        if isinstance(self.env.robot_uids, str):
            robots = [self.env.robot_uids]
        else:
            robots = self.env.robot_uids

        self.current_pos_per_robot = [0, 1]
        self.available_slot = self.original_spawn_slot
            

        for i, robot in enumerate(robots):
            pose = robot_slot.pop(0)
            if robot == "panda":
                self.env.agent.agents[i].reset(panda_qpos)
                self.env.agent.agents[i].robot.set_pose(pose)
            else:
                raise NotImplementedError(f"Not supported robot uid : {self.env.robot_uids}")
            
    def compute_random_pose(self, nb : int):
        # Simple implementation, wont work if you increase the number of objects or the number of type
        random.shuffle(self.available_slot)
        out = []
        if nb > 1:
            for slot in self.available_slot:
                if len(slot) != 4:
                    for i in range(nb):
                        out.append(slot.pop(i))
                    break
        else:
            out = []
            for slot in self.available_slot:
                if len(slot) != 0:
                    out = [torch.tensor(slot.pop(0))]
                    break
        if len(out) == 0:
            raise RuntimeError("Not found any space left")
        return out
            
    def move_robot_to_slot(self, agent_idx : int, hall_idx : int, env_id : int) -> str:

        # Can be optimized, not create at each time the torch tensor or batching modif (in the runner)
        # to apply all in one

        if agent_idx < 0 or agent_idx > len(self.env.agent.agents):
            raise RuntimeError(f"Invalid agent_idx : {agent_idx}")
        
        if hall_idx < 0 or hall_idx > 4:
            raise RuntimeError(f"Invalid hall_idx : {hall_idx}")
        
        for agent_id, agent_slot in enumerate(self.current_pos_per_robot):
            if agent_slot == hall_idx and agent_id != agent_idx:
                return "Already occuped"
        
        robot = self.env.agent.agents[agent_idx].robot

        # --- Get batched pose ---
        batched_pose = robot.get_pose()          # sapien.Pose (batched)
        raw_pose = batched_pose.raw_pose.clone() # (B, 7) torch tensor

        # --- Overwrite only env_id ---
        slot_pose = self.robot_slot[hall_idx]    # sapien.Pose (single)

        raw_pose[env_id, 0:3] = torch.Tensor(slot_pose.p)
        raw_pose[env_id, 3:7] = torch.Tensor(slot_pose.q)

        # --- Push back ---
        robot.set_pose(Pose(raw_pose=raw_pose))

        return ""
    
    def get_default_obs(self) -> dict:
        return self._default_obs
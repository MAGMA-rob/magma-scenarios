import os.path as osp
from pathlib import Path
import random

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.structs import Pose

table_height = 0.52
table_lenght = 1.38
table_width = 0.69
GRID_STEP = 0.1
GAP = 0.15
class MultipleHallSceneBuilder(SceneBuilder):
    """
    Scene builder for the four-hall sorting environment.
    Each hall contains one table, one robot slot, and per-robot trays.
    """

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

    def build(self, n: int):
        if n > 4:
            raise TypeError("Trying to build a scene with more than 4 hall. Not currently supported.")

        model_dir = Path(osp.dirname(__file__)) / "assets"
        table_model_file = str(model_dir / "table.glb")

        self.halls = []
        self.trays = []
        self.robot_slot = []
        self.current_pos_per_robot = []
        self.original_spawn_slot = []
        self.available_slot = []
        self._default_obs = {}

        table_positions = [
            [-2.0, 0.0],   # left
            [0.0, 2.0],    # top
            [2.0, 0.0],    # right
            [0.0, -2.0],   # bottom
            
        ]

        robot_positions = [
            [-2.3, -0.3],   # near left table
            [-0.3, 1.7],    # near top table
            [1.7, -0.3],    # near right table
            [-0.3, -2.3],   # near bottom table
            
        ]

        for i in range(n):
            table_x, table_y = table_positions[i]
            robot_x, robot_y = robot_positions[i]

            builder = self.scene.create_actor_builder()

            builder.add_box_collision(
                pose=sapien.Pose(p=[0, 0, table_height / 2]),
                half_size=(table_lenght / 2, table_width / 2, table_height / 2),
            )

            builder.add_visual_from_file(
                filename=table_model_file,
                scale=[1, 1, 1],
                pose=sapien.Pose(
                    p=[0, 0, 0],
                    q=euler2quat(0, 0, np.pi / 2),
                ),
            )

            builder.initial_pose = sapien.Pose(
                p=[table_x, table_y, -table_height],
                q=euler2quat(0, 0, 0),
            )

            table = builder.build_kinematic(name=f"table-workspace-{i}")
            self.halls.append(table)

            self.robot_slot.append(
                sapien.Pose(
                    p=[robot_x, robot_y, 0],
                    q=euler2quat(0, 0, 0),
                )
            )

            self._default_obs[f"Hall{i+1}"] = torch.tensor(
                [table_x + GAP, table_y , table_height, 1, 0, 0, 0]
            )

            hall_slot = [
                [table_x + GRID_STEP + GAP, table_y + GRID_STEP, 0.02],
                [table_x + GRID_STEP + GAP, table_y - GRID_STEP, 0.02],
                [table_x - GRID_STEP + GAP, table_y + GRID_STEP, 0.02],
                [table_x - GRID_STEP + GAP, table_y - GRID_STEP, 0.02],
            ]
            self.original_spawn_slot.append(hall_slot)

        if isinstance(self.env.robot_uids, str):
            robots = [self.env.robot_uids]
        else:
            robots = self.env.robot_uids

        for i, _ in enumerate(robots):
            self.trays.append(
                actors.build_box(
                    scene=self.scene,
                    half_sizes=np.array([0.16, 0.16, 0.005], dtype=np.float32),
                    color=np.array([1, 1, 0, 1], dtype=np.float32),
                    name=f"tray-{i}",
                    body_type="kinematic",
                    initial_pose=sapien.Pose(p=[0, 0, 0.08]),
                )
            )

        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500

        self.ground = build_ground(
            self.scene,
            floor_width=floor_width,
            altitude=-table_height,
        )

        self.scene_objects = [*self.halls, *self.trays, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        if isinstance(self.env.robot_uids, str):
            robots = [self.env.robot_uids]
        else:
            robots = self.env.robot_uids

        self.current_pos_per_robot = []
        self.available_slot = [slot.copy() for slot in self.original_spawn_slot]

        for i, robot in enumerate(robots):
            slot_idx = i
            pose = self.robot_slot[slot_idx]

            if robot == "panda":
                self.env.agent.agents[i].reset(self.panda_qpos)
                self.env.agent.agents[i].robot.set_pose(pose)
                self._move_tray_to_robot_slot(i, slot_idx)
                self.current_pos_per_robot.append(slot_idx)
            else:
                raise NotImplementedError(f"Not supported robot uid : {self.env.robot_uids}")

    def compute_random_pose(self, nb: int):
        random.shuffle(self.available_slot)

        out = []

        for slot in self.available_slot:
            if len(slot) >= nb:
                for _ in range(nb):
                    out.append(torch.tensor(slot.pop(0)))
                break

        if len(out) == 0:
            raise RuntimeError("Not found any space left")

        return out

    def move_robot_to_slot(self, agent_idx: int, hall_idx: int, env_id: int) -> str:
        if agent_idx < 0 or agent_idx >= len(self.env.agent.agents):
            raise RuntimeError(f"Invalid agent_idx : {agent_idx}")

        if hall_idx < 0 or hall_idx >= len(self.robot_slot):
            raise RuntimeError(f"Invalid hall_idx : {hall_idx}")

        for other_agent_idx, current_slot in enumerate(self.current_pos_per_robot):
            if current_slot == hall_idx and other_agent_idx != agent_idx:
                return "Already occupied"

        robot = self.env.agent.agents[agent_idx].robot
        slot_pose = self.robot_slot[hall_idx]

        raw_pose = robot.get_pose().raw_pose.clone()
        raw_pose[env_id, 0:3] = torch.tensor(slot_pose.p, device=raw_pose.device)
        raw_pose[env_id, 3:7] = torch.tensor(slot_pose.q, device=raw_pose.device)

        robot.set_pose(Pose(raw_pose=raw_pose))

        self.current_pos_per_robot[agent_idx] = hall_idx
        self._move_tray_to_robot_slot(agent_idx, hall_idx, env_id)

        return ""

    def _move_tray_to_robot_slot(self, agent_idx: int, slot_idx: int, env_id: int | None = None):
        robot_pose = self.robot_slot[slot_idx]

        tray_p = np.array(robot_pose.p, dtype=np.float32) + np.array([0.0, 0.55, 0.1])
        tray_q = robot_pose.q

        tray = self.trays[agent_idx]

        if env_id is None:
            tray.set_pose(sapien.Pose(p=tray_p, q=tray_q))
            return

        raw_pose = tray.pose.raw_pose.clone()
        raw_pose[env_id, 0:3] = torch.tensor(tray_p, device=raw_pose.device)
        raw_pose[env_id, 3:7] = torch.tensor(tray_q, device=raw_pose.device)
        tray.set_pose(Pose(raw_pose=raw_pose))

    def get_default_obs(self) -> dict:
        return self._default_obs
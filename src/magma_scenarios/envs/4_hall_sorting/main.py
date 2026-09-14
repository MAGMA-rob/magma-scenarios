from typing import Dict, Tuple
import random

import sapien
import torch
from pathlib import Path
from magma_scenarios.envs.visual_assets import OBJECT_VISUALS
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig

from magma_core.simulation.envs import DefaultMultiAgentEnv
from magma_scenarios.scene_builders import MultipleHallSceneBuilder


@register_env("4HallSorting-v1", max_episode_steps=200)
class HallSorting(DefaultMultiAgentEnv):
    """
    Four-hall sorting environment.

    options example:
    {
        "Hall1": ["book_1", "pen_2"],
        "Hall3": ["backpack_1", "package_4"],
    }

    If no hall assignment is provided, objects are randomized across halls.
    """

    SUPPORTED_ROBOTS = [("panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda]]

    hall_names = ["Hall1", "Hall2", "Hall3", "Hall4"]

    object_specs = {
        "book": {
            "half_size": (0.035, 0.025, 0.02),
            "color": (0.1, 0.25, 1.0, 1.0),
        },
        "pen": {
            "half_size": (0.04, 0.01, 0.015),
            "color": (1.0, 0.1, 0.1, 1.0),
        },
        "backpack": {
            "half_size": (0.02, 0.02, 0.025),
            "color": (0.1, 0.65, 0.2, 1.0),
            "visual": "eraser",
        },
        "package": {
            "half_size": (0.015, 0.015, 0.025),
            "color": (0.75, 0.45, 0.15, 1.0),
            "visual": "sharpener",
        },
    }

    def __init__(self,*args,robot_uids=("panda", "panda"),robot_init_qpos_noise=0.0,**kwargs):
        super().__init__(*args,robot_uids=robot_uids,robot_init_qpos_noise=robot_init_qpos_noise,**kwargs,)

    def _load_agent(self, options: dict, initial_agent_poses=None):
        super()._load_agent(options,[sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])])

    def _load_scene(self, options: dict):
        self._load_layout_options(options)

        self.table_scene = MultipleHallSceneBuilder(env=self)
        self.table_scene.build(4)

        selected_objects = self._select_object_names()

        self.objects = []
        self.object_by_name = {}

        for name in selected_objects:
            obj_type = self._object_type_from_name(name)
            obj = self._build_object(name, obj_type)
            self.objects.append(obj)
            self.object_by_name[name] = obj

    @property
    def _default_human_render_camera_configs(self):
        # this is just like _sensor_configs, but for adding cameras used for rendering when you call env.render()
        # when render_mode="rgb_array" or env.render_rgb_array()
        # Another feature here is that if there is a camera called render_camera, this is the default view shown initially when a GUI is opened
        pose = sapien_utils.look_at([0, 6, 3], [0.0, 0., 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _select_object_names(self):
        requested_objects = []

        for hall in self.hall_names:
            requested_objects.extend(self.hall_assignment.get(hall, []))

        if requested_objects:
            return requested_objects

        return [
            f"{obj_type}_{idx}"
            for obj_type in self.object_specs
            for idx in range(1, 5)
        ]

    def _object_type_from_name(self, name: str):
        for obj_type in self.object_specs:
            if name.startswith(f"{obj_type}_"):
                return obj_type

        raise ValueError(f"Unknown object type for {name}.")

    def _load_layout_options(self, options: dict):
        self.hall_assignment = {
            hall: options[hall]
            for hall in self.hall_names
            if hall in options
        }

    def _build_object(self, name: str, obj_type: str):
        spec = self.object_specs[obj_type]

        return self.build_box_object(
            name=name,
            half_size=spec["half_size"],
            color=spec["color"],
            visual=OBJECT_VISUALS.get(spec.get("visual", obj_type)),
            visual_assets_dir=(
                Path(__file__).resolve().parents[2]
                / "assets"
                / "visuel"
            ),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._load_layout_options(options)

        with torch.device(self.device):
            self.table_scene.initialize(env_idx)

        batch_size = len(env_idx)
        self._place_objects(batch_size)

    def _place_objects(self, batch_size: int):
        assignment = self._resolved_assignment()
        available_cells = self._hall_cells()

        placed = set()
        q = [1, 0, 0, 0]

        for hall_name, object_names in assignment.items():
            cells = available_cells[hall_name]

            for object_name in object_names:
                if object_name in placed:
                    raise ValueError(f"{object_name} is assigned more than once.")

                if object_name not in self.object_by_name:
                    raise ValueError(f"Unknown object name: {object_name}.")

                if len(cells) == 0:
                    raise RuntimeError(f"No free cell left in {hall_name}.")

                cell = cells.pop(0)
                obj = self.object_by_name[object_name]
                self._place_object(obj, cell, batch_size, q)
                placed.add(object_name)

    def _resolved_assignment(self):
        if self.hall_assignment:
            return self.hall_assignment

        object_names = list(self.object_by_name.keys())
        random.shuffle(object_names)

        assignment = {
            "Hall1": [],
            "Hall2": [],
            "Hall3": [],
            "Hall4": [],
        }

        for i, object_name in enumerate(object_names):
            hall = self.hall_names[i % len(self.hall_names)]
            assignment[hall].append(object_name)

        return assignment


    def _hall_cells(self):
        cells = {}

        for hall_idx, hall_name in enumerate(self.hall_names):
            raw_cells = self.table_scene.original_spawn_slot[hall_idx]
            cells[hall_name] = [list(cell) for cell in raw_cells]
            random.shuffle(cells[hall_name])

        return cells

    def _place_object(self, obj, cell, batch_size: int, q):

        xyz = torch.tensor(
            [cell[0], cell[1], cell[2]],
            device=self.device,
            dtype=torch.float32,
        ).repeat(batch_size, 1)

        obj.set_pose(Pose.create_from_pq(p=xyz, q=q))

    def _get_obs_extra(self, info: dict):
        obs = {}

        for hall_name, pose in self._hall_obs().items():
            obs[hall_name] = pose

        for tray in self.table_scene.trays:
            obs[tray.name] = tray.pose.raw_pose

        for obj in self.objects:
            obs[obj.name] = obj.pose.raw_pose

        return obs

    def _hall_obs(self):
        obs = {}
        builder_obs = self.table_scene.get_default_obs()

        for hall_name in self.hall_names:
            hall_pose = builder_obs[hall_name].to(
            device=self.device,
            dtype=torch.float32,
        )
            if hall_pose.ndim == 1:
                hall_pose = hall_pose.repeat(self.num_envs, 1)

            obs[hall_name] = hall_pose

        return obs

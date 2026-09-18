from typing import Dict
from pathlib import Path
from magma_scenarios.envs.visual_assets import OBJECT_VISUALS
import numpy as np
import random
import sapien
import torch
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from math import pi
from transforms3d.euler import euler2quat
from magma_scenarios.envs.asset_lib import create_trashcan
from magma_scenarios.scenarios.table_cleaning.common.attributes import (
    CLEAN_STATE,
    DIRTY_STATE,
    STATIC_STATE,
    cleaning_objects,
    dishware,
    food,
)
from magma_scenarios.scenarios.table_cleaning.common.layout import (
    DISH_STORAGE_CENTER,
    FOOD_STORAGE_CENTER,
    STORAGE_OFFSETS,
    TABLE_CENTER,
    TABLE_OFFSETS,
)

BLUE = (0.0, 0.2, 1.0, 1.0)
BLUE_LIGHT = (0.45, 0.55, 1.0, 1.0)
RED = (1, 0, 0, 1)
RED_LIGHT = (1.0, 0.7, 0.7, 1.0)
YELLOW = (1, 1, 0, 1)

CLEAN_STATE = 1
DIRTY_STATE = 0
STATIC_STATE = -1

class CleanTableCommonMixin:
    table_center = list(TABLE_CENTER)
    food_storage_centre = list(FOOD_STORAGE_CENTER)
    dish_storage_centre = list(DISH_STORAGE_CENTER)
    trash_pose = [-1, 0.5, 0]

    food_pool = food
    dish_pool = dishware

    def _load_common_scene(self, options: dict):
        self._load_layout_options(options)

        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

        self.food_storage = self._build_storage_surface(
            "food_storage",
            self.food_storage_centre,
            BLUE_LIGHT,
        )
        self.dish_storage = self._build_storage_surface(
            "dish_storage",
            self.dish_storage_centre,
            RED_LIGHT,
        )

        selected_food, selected_dish = self._select_object_names()
        self.all_objects = selected_food + selected_dish

        self.food = [self._build_object(name, BLUE) for name in selected_food]
        self.dishware = [self._build_object(name, RED) for name in selected_dish]
        self.cleaning_objects = [
            self._build_object(
                name=name,
                color=YELLOW,
                half_size=(0.04, 0.03, 0.013),
                initial_z=0.02,
            )
            for name in cleaning_objects
        ]

        self.trash = create_trashcan(self.scene, add_collision=False)
        trash_catcher_builder = self.scene.create_actor_builder()
        trash_catcher_builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.01]),
            half_size=[0.09, 0.09, 0.01],
        )
        trash_catcher_builder.set_initial_pose(sapien.Pose(p=self.trash_pose))
        self.trash_catcher = trash_catcher_builder.build_kinematic(
            name="trash_catcher"
        )

    def _load_layout_options(self, options: dict):
        self.obj_on_table = options.get("table", [])
        self.obj_on_storage = options.get("storage", [])
        self.dirty_obj = options.get("dirty", [])

    def _select_object_names(self):
        requested_objects = self.obj_on_storage + self.obj_on_table
        if not requested_objects:
            return random.sample(self.food_pool, k=2), random.sample(self.dish_pool, k=3)

        selected_food = [obj for obj in requested_objects if obj in self.food_pool]
        selected_dish = [obj for obj in requested_objects if obj in self.dish_pool]
        return selected_food, selected_dish

    def _build_storage_surface(self, name, center, color, h = 0.005):
        return actors.build_box(
            scene=self.scene,
            half_sizes=np.array([0.08, 0.16, h], dtype=np.float32),
            color=color,
            name=name,
            body_type="static",
            initial_pose=sapien.Pose(p=[center[0], center[1], 0.005]),
        )

    def _build_collision_box(self, name, center, size):
        return self.create_box(
            thickness=0.01,
            size=size,
            height=0.1,
            name=name,
            initial_pose=np.array(center),
            add_bottom_wall=True,
        )

    def _build_object(
            self,
            name: str,
            color,
            half_size=(0.02, 0.02, 0.02),
            initial_z: float | None = None,
        ):
        name_parts = name.rsplit("_", 1)

        if (len(name_parts) == 2 and name_parts[1].isdigit()):
            object_type = name_parts[0]
        else:
            object_type = name

        visual_key = object_type

        dirty_visual_key = f"dirty_{object_type}"

        if (name in self.dirty_obj and dirty_visual_key in OBJECT_VISUALS):
            visual_key = dirty_visual_key

        return self.build_box_object(
            name=name,
            half_size=half_size,
            color=color,
            initial_z=initial_z,
            visual=OBJECT_VISUALS.get(visual_key),
            visual_assets_dir=(
                Path(__file__).resolve().parents[2]
                / "assets"
                / "visuel"
            ),
        )


    def _initialize_common_fixed_actors(self):
        self.trash.set_pose(
            sapien.Pose(
                p=self.trash_pose,
                q=euler2quat(0, 0, -pi),
            )
        )
        self.trash_catcher.set_pose(sapien.Pose(p=self.trash_pose))
        self._set_first_joint(self.trash, pi)
        
    def _initialize_common_episode(self, env_idx: torch.Tensor):
        batch_size = len(env_idx)
        q = [1, 0, 0, 0]

        table_cells = self._cells_around(self.table_center, TABLE_OFFSETS)
        food_storage_cells = self._cells_around(self.food_storage_centre, STORAGE_OFFSETS)
        dish_storage_cells = self._cells_around(self.dish_storage_centre, STORAGE_OFFSETS)

        self.table_scene.initialize(env_idx)
        self.objects = []
        self.object_states = self._default_object_states()

        self._sample_default_layout_if_needed()

        self._place_objects(self.food, food_storage_cells, table_cells, batch_size, q)
        self._place_objects(self.dishware, dish_storage_cells, table_cells, batch_size, q)

        self._apply_dirty_states()
        self._place_cleaning_objects(table_cells, batch_size, q)

    def _cells_around(self, center, offsets):
        return [(center[0] + dx, center[1] + dy) for dx, dy in offsets]

    def _default_object_states(self):
        return {
            obj.name: torch.full(
                (self.num_envs,),
                CLEAN_STATE,
                dtype=torch.int32,
                device=self.device,
            )
            for obj in [*self.food, *self.dishware, *self.cleaning_objects]
        }

    def get_magma_extra_state(self):
        if not hasattr(self, "object_states"):
            return {}

        return {
            name: state.clone()
            for name, state in self.object_states.items()
        }

    def set_magma_extra_state(self, state: dict):
        if not hasattr(self, "object_states"):
            return

        for name, value in state.items():
            if name in self.object_states:
                self.object_states[name] = value.clone().to(device=self.device)

    def _sample_default_layout_if_needed(self):
        if self.obj_on_table or self.obj_on_storage:
            return

        self.obj_on_table = random.sample(self.all_objects, k=len(self.all_objects) // 2)
        self.obj_on_storage = [
            obj for obj in self.all_objects
            if obj not in self.obj_on_table
        ]

    def _place_objects(self, objects, storage_cells, table_cells, batch_size, q):
        for obj in objects:
            if obj.name in self.obj_on_table:
                self._place_object(obj, table_cells, z=0, batch_size=batch_size, q=q)
            elif obj.name in self.obj_on_storage:
                self._place_object(obj, storage_cells, z=0.01, batch_size=batch_size, q=q)

    def _place_cleaning_objects(self, table_cells, batch_size, q):
        for obj in self.cleaning_objects:
            self.object_states[obj.name][:] = STATIC_STATE
            self._place_object(obj, table_cells, z=0, batch_size=batch_size, q=q)

    def _place_object(self, obj, cells, z, batch_size, q):
        cell = self._pop_random_cell(cells)
        xyz = torch.tensor(
            [cell[0], cell[1], z],
            device=self.device,
            dtype=torch.float32,
        ).repeat(batch_size, 1)

        obj.set_pose(Pose.create_from_pq(p=xyz, q=q))

        if obj not in self.objects:
            self.objects.append(obj)

    def _pop_random_cell(self, cells):
        random_index = torch.randint(0, len(cells), (1,)).item()
        return cells.pop(random_index)

    def _apply_dirty_states(self):
        for obj in self.objects:
            self.object_states[obj.name][:] = (
                DIRTY_STATE if obj.name in self.dirty_obj else CLEAN_STATE
            )

    def _add_common_static_obs(self, obs: Dict):
        static_entries = [
            (self.trash.name, self.trash.pose.raw_pose, {}),
            (self.dish_storage.name, self.dish_storage.pose.raw_pose, {}),
            (self.food_storage.name, self.food_storage.pose.raw_pose, {}),
            ("table", self._table_pose(), {}),
        ]

        for name, pose, extra in static_entries:
            self._add_static_obs(obs, name, pose, **extra)

    def _add_object_obs(self, obs: Dict):
        for obj in self.objects:
            obs[obj.name] = {
                "pose": obj.pose.raw_pose,
                "state": self.object_states[obj.name],
            }

    def _add_static_obs(self, obs: Dict, name: str, pose, **extra):
        obs[name] = {
            "pose": pose,
            "state": self._static_state(),
            **extra,
        }

    def _static_state(self):
        return torch.full(
            (self.num_envs,),
            STATIC_STATE,
            dtype=torch.int32,
            device=self.device,
        )

    def _table_pose(self):
        return torch.tensor(
            [self.table_center[0], self.table_center[1], 0.0, 1.0, 0.0, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.num_envs, 1)

    def _set_first_joint(self, actor, value: float):
        qpos = actor.get_qpos()
        qpos[0] = value
        actor.set_qpos(qpos)

    @property
    def _default_human_render_camera_configs(self):
        # this is just like _sensor_configs, but for adding cameras used for rendering when you call env.render()
        # when render_mode="rgb_array" or env.render_rgb_array()
        # Another feature here is that if there is a camera called render_camera, this is the default view shown initially when a GUI is opened
        pose = sapien_utils.look_at([1, 0, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

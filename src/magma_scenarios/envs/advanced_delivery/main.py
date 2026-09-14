import inspect
import os.path as osp
import random
from pathlib import Path
from typing import Tuple
from magma_scenarios.envs.visual_assets import OBJECT_VISUALS
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from magma_core.simulation.envs import DefaultMultiAgentEnv
from magma_scenarios.envs.asset_lib import create_trashcan


CLEAN_STATE = 1
BROKEN_STATE = 0
STATIC_STATE = -1

PACKAGE_FREE = 0
TRAY_RECEPTION_ROBOT = 1
PACKAGE_PREPARATION_A = 2
PACKAGE_PREPARATION_B = 3


PRODUCT_TYPES = [
    "electronics",
    "drinks",
    "snacks",
    "hygiene",
    "textile",
]

CATEGORY_STORAGE_TABLE = {
    "electronics": "storage_a",
    "drinks": "storage_a",
    "snacks": "storage_a",
    "hygiene": "storage_b",
    "textile": "storage_b",
}


@register_env("MultiRobotDelivery-v1", max_episode_steps=300)
class MultiRobotDeliveryEnv(DefaultMultiAgentEnv):
    SUPPORTED_ROBOTS = [("panda", "panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda, Panda]]

    object_specs = {
        "electronics": {
            "half_size": (0.02, 0.02, 0.02),
            "color": [0.05, 0.15, 0.90, 1.0],
            "body_type": "dynamic",
            "visual": "electronic",
        },
        "drinks": {
            "half_size": (0.02, 0.02, 0.02),
            "color": [0.05, 0.70, 0.95, 1.0],
            "body_type": "dynamic",
            "visual": "drink",
        },
        "snacks": {
            "half_size": (0.02, 0.02, 0.02),
            "color": [0.95, 0.45, 0.00, 1.0],
            "body_type": "dynamic",
            "visual": "snack",
        },
        "hygiene": {
            "half_size": (0.02, 0.02, 0.02),
            "color": [0.35, 0.75, 0.05, 1.0],
            "body_type": "dynamic",
        },
        "textile": {
            "half_size": (0.02, 0.02, 0.02),
            "color": [0.95, 0.05, 0.15, 1.0],
            "body_type": "dynamic",
        },
        "priority": {
            "half_size": (0.15, 0.15, 0.01),
            "color": [0.95, 0.20, 0.20, 1.0],
            "body_type": "kinematic",
        },
        "standard": {
            "half_size": (0.15, 0.15, 0.01),
            "color": [0.25, 0.35, 0.95, 1.0],
            "body_type": "kinematic",
        },
        "reception_helper_tray": {
            "half_size": (0.15, 0.15, 0.01),
            "color": [0.95, 0.85, 0.10, 1.0],
            "body_type": "kinematic",
        },
    }
    cube_half_size = 0.02
    tray_height = 0.01

    category_grid_r = 0.85
    
    table_height = 0.9196429
    table_length = 2.418
    table_width = 1.209
    table_scale = 1.75

    reception_grid_r = 0.1
    storage_grid_r = 0.12
    package_grid_r = 0.07

    max_instances_per_product = 8
    max_active_per_product = 4

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

    def __init__(
        self,
        *args,
        robot_uids=("panda", "panda", "panda"),
        robot_init_qpos_noise=0.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs,
        )

    def _load_agent(self, options: dict, initial_agent_poses=None):
        super()._load_agent(
            options,
            [
                sapien.Pose(p=[-2.55, -1.55, 0.0], q=euler2quat(0, 0, 0)),
                sapien.Pose(p=[0.25, 1.55, 0.0], q=euler2quat(0, 0, 0)),
                sapien.Pose(p=[2.10, -0.55, 0.0], q=euler2quat(0, 0, np.pi)),
            ],
        )

    def _load_scene(self, options: dict):
        self._load_layout_options(options)

        self.table_centers = {
            "output": np.array([-1.55, 1.65], dtype=np.float32),
            "storage_a": np.array([1.15, 0.75], dtype=np.float32),
            "storage_b": np.array([1.15, -0.75], dtype=np.float32),
            "reception": np.array([-1.55, -1.65], dtype=np.float32),
        }

        self.tables = {
            "output_table": self._build_table("output_table", self.table_centers["output"], yaw=0.0),
            "storage_table_a": self._build_table("storage_table_a", self.table_centers["storage_a"], yaw=np.pi / 2),
            "storage_table_b": self._build_table("storage_table_b", self.table_centers["storage_b"], yaw=np.pi / 2),
            "reception_table": self._build_table("reception_table", self.table_centers["reception"], yaw=0.0),
        }

        self.ground = build_ground(self.scene, floor_width=100, altitude=-self.table_height)

        self.object_slots = self._make_object_slots()
        self.robot_slots = self._make_robot_slots()
        self.package_grid_offsets = self._grid_offsets(rows=2, cols=2, r=self.package_grid_r)

        self.trashcan = create_trashcan(self.scene, name="trashcan")
        self.trash_collision = self.create_box(
            thickness=0.01,
            size=0.2,
            height=0.1,
            name="washing_machine_basket",
            initial_pose=np.array((0,0,0)),
            add_bottom_wall=True
        )
        self.reception_helper_tray = self._build_object(
            "reception_helper_tray",
            "reception_helper_tray"
        )

    
        self.packages = [
        self._build_object("priority_0","priority"),  # priority
        self._build_object("priority_1","priority"),  # priority
        self._build_object("standard_0", "standard"),  # standard
        self._build_object("standard_1","standard"),  # standard
    ]

        self.episode_product_specs = self._build_episode_specs()
        self.product_objects = self._build_episode_products(self.episode_product_specs)

    def _build_episode_products(self, specs):
        product_names = specs["reception"] + specs["storage"]

        return [
            self._build_object(
                name,
                self._product_type_from_name(name),
            )
            for name in product_names
        ]

    def _load_layout_options(self, options: dict):
        options = options or {}

        self.option_reception_products = options.get("products_in_reception", [])
        self.option_broken_products = options.get("broken_products", [])
        self.option_storage_products = options.get("products_in_storage", [])

    def _build_table(self, name: str, center_xy, yaw: float):
        builder = self.scene.create_actor_builder()

        model_dir = Path(osp.dirname(inspect.getfile(TableSceneBuilder))) / "assets"
        table_model_file = str(model_dir / "table.glb")

        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, self.table_height / 2]),
            half_size=(self.table_length / 2, self.table_width / 2, self.table_height / 2),
        )
        builder.add_visual_from_file(
            filename=table_model_file,
            scale=[self.table_scale] * 3,
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        )
        builder.initial_pose = sapien.Pose(
            p=[center_xy[0], center_xy[1], -self.table_height],
            q=euler2quat(0, 0, yaw),
        )
        return builder.build_kinematic(name=name)

    def _build_object(self, name: str, obj_type: str):
        spec = self.object_specs[obj_type]

        return self.build_box_object(
            name=name,
            half_size=spec["half_size"],
            color=spec["color"],
            body_type=spec["body_type"],
            initial_xy=(8, 8),
            visual=OBJECT_VISUALS.get(spec.get("visual", obj_type)),
            visual_assets_dir=(
                Path(__file__).resolve().parents[2]
                / "assets/visuel"
            ),
        )

    def _grid_offsets(self, rows: int, cols: int, r: float):
        x0 = -r * (cols - 1) / 2
        y0 = -r * (rows - 1) / 2
        return [
            (x0 + col * r, y0 + row * r)
            for row in range(rows)
            for col in range(cols)
        ]

    def _grid_cells(self, center_xy, rows: int, cols: int, r: float, z: float):
        return [
            [center_xy[0] + dx, center_xy[1] + dy, z]
            for dx, dy in self._grid_offsets(rows, cols, r)
        ]

    def _make_object_slots(self):
        c = self.table_centers
        self.reception_grid_center = self.table_centers["reception"] + np.array([-0.3, 0.0])
        return {
            "reception_returns": self._grid_cells(
                self.reception_grid_center, rows=3, cols=3, r=self.reception_grid_r, z=self.cube_half_size
            ),
            "output_packages": [
                [c["output"][0] - 0.75, c["output"][1], self.tray_height],
                [c["output"][0] - 0.25, c["output"][1], self.tray_height],
                [c["output"][0] + 0.25, c["output"][1], self.tray_height],
                [c["output"][0] + 0.75, c["output"][1], self.tray_height],
            ],
            "storage_a_products": self._make_storage_product_grids("storage_a"),
            "storage_b_products": self._make_storage_product_grids("storage_b"),
        }

    def _make_storage_product_grids(self, table_name: str):
        """
        {
            "electronics": [
                [x1, y1, z],
                [x2, y2, z],
                [x3, y3, z],
                [x4, y4, z],
            ],
            "drinks": [
                [x1, y1, z],
                [x2, y2, z],
                [x3, y3, z],
                [x4, y4, z],
            ],
            "snacks": [
                [x1, y1, z],
                [x2, y2, z],
                [x3, y3, z],
                [x4, y4, z],
            ],
        }
        """
        grids = {}
        centers = self._product_grid_centers()

        for product_type, center_xy in centers.items():
            if CATEGORY_STORAGE_TABLE[product_type] != table_name:
                continue

            grids[product_type] = self._grid_cells(
                center_xy,
                rows=2,
                cols=2,
                r=self.storage_grid_r,
                z=self.cube_half_size,
            )

        return grids

    def _product_grid_centers(self):
        x = self.table_centers["storage_a"][0]
        y0 = 0.0
        r = self.category_grid_r

        return {
            "electronics": [x, y0 + 2 * r],
            "drinks": [x, y0 + 1 * r],
            "snacks": [x, y0],
            "hygiene": [x, y0 - 1 * r],
            "textile": [x, y0 - 2 * r],
        }

    def _product_robot_slot_specs(self):
        r = self.category_grid_r

        return {
            "electronics": ("storage_a", "left", 0),
            "drinks": ("storage_a", "right", 1),
            "snacks": ("storage_a", "left", 2),
            "hygiene": ("storage_b", "right", 3),
            "textile": ("storage_b", "left", 4),
        }



    def _make_robot_slots(self):
        """
        Storage access slots follow one continuous zigzag over both storage tables:
        storage_a_left_0   -> electronics_spot
        storage_a_right_1  -> drinks_spot
        storage_a_left_2   -> snacks_spot
        storage_b_right_3  -> hygiene_spot
        storage_b_left_4   -> stationery_spot
        storage_b_right_5  -> textile_spot
        """

        c = self.table_centers
        slots = {
            "reception": sapien.Pose(
                p=[c["reception"][0] - 0.90, c["reception"][1] + 0.20, 0.0],
                q=euler2quat(0, 0, 0),
            )
        }

        output_xs = [
            c["output"][0] - 0.75,
            c["output"][0] - 0.25,
            c["output"][0] + 0.25,
            c["output"][0] + 0.75,
        ]
        for i, x in enumerate(output_xs):
            side = -1 if i % 2 == 0 else 1
            slots[f"output_{i}"] = sapien.Pose(
                p=[x, c["output"][1] + side * 0.85, 0.0],
                q=euler2quat(0, 0, -side * np.pi / 2),
            )


        product_centers = self._product_grid_centers()

        for product_type, (table_name, side_name, idx) in self._product_robot_slot_specs().items():
            center = c[table_name]
            product_center = product_centers[product_type]

            is_left = side_name == "left"
            x = center[0] - 0.5 if is_left else center[0] + 0.5
            yaw = 0 if is_left else np.pi

            slots[f"{table_name}_{side_name}_{idx}"] = sapien.Pose(
                p=[x, product_center[1], 0.0],
                q=euler2quat(0, 0, yaw),
            )
                    

        return slots

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._load_layout_options(options)

        with torch.device(self.device):
            b = len(env_idx)

            self.objects = []
            self.object_states = {}

            self.package_states = {
                package.name: torch.full(
                    (self.num_envs,),
                    PACKAGE_FREE,
                    dtype=torch.int32,
                    device=self.device,
                )
                for package in self.packages
            }

            self.package_states["reception_helper_tray"] = torch.full(
                (self.num_envs,),
                TRAY_RECEPTION_ROBOT,
                dtype=torch.int32,
                device=self.device,
            )

            self._initialize_robots()
            self._initialize_fixed_actors()
            self._initialize_packages(b)
            self._place_episode_products(b)


    def _initialize_robots(self):
        self.current_robot_slots = [
            "reception",
            "storage_a_left_0",
            "storage_b_left_4",
        ]
        self._reset_robot(0, "reception")
        self._reset_robot(1, "storage_a_left_0")
        self._reset_robot(2, "storage_b_left_4")

    def _initialize_fixed_actors(self):
        trash_xyz = [-3.1, -1.7, 0.0]

        self.trashcan.set_pose(
            sapien.Pose(
                p=trash_xyz,
                q=euler2quat(0, 0, -np.pi),
            )
        )
        self._set_first_joint(self.trashcan, np.pi)
        self.trash_collision.set_pose(
            sapien.Pose(
                p=trash_xyz,
                q=euler2quat(0, 0, 0),
            )
        )
        self.reception_helper_tray.set_pose(
            sapien.Pose(
                p=[-2.55, -0.9, 0.02],
                q=euler2quat(0, 0, 0),
            )
        )
        
    def _set_first_joint(self, actor, value: float):
        qpos = actor.get_qpos()
        qpos[0] = value
        actor.set_qpos(qpos)
    
    def _initialize_packages(self, batch_size: int):
        for package, xyz in zip(self.packages, self.object_slots["output_packages"]):
            package.set_pose(self._pose_from_xyz(xyz, batch_size))

    def _build_episode_specs(self):
        if (
            not self.option_reception_products
            and not self.option_storage_products
            and not self.option_broken_products
        ):
            return self._random_episode_specs()

        return {
            "reception": list(self.option_reception_products),
            "storage": list(self.option_storage_products),
            "broken": set(self.option_broken_products),
        }

    def _random_episode_specs(self):
        """
        {
        "reception": [
            "electronics",
            "drinks",
            "textile",
            "snacks",
        ],
        "storage_a": [
            "electronics",
            "electronics",
            "drinks",
            "snacks",
        ],
        "storage_b": [
            "hygiene",
            "stationery",
            "textile",
            "textile",
        ],
        }
        """
        specs = {
            "reception": [],
            "storage": [],
            "broken": set(),
        }

        reception_capacity = len(self.object_slots["reception_returns"])
        remaining_reception_capacity = reception_capacity

        for product_type in PRODUCT_TYPES:
            count = random.randint(1, self.max_active_per_product)

            product_names = [
                f"{product_type}_{i}"
                for i in range(count)
            ]

            max_reception_for_type = min(2, count, remaining_reception_capacity)
            reception_count = random.randint(0, max_reception_for_type)

            reception_products = random.sample(product_names, k=reception_count)
            storage_products = [
                product_name
                for product_name in product_names
                if product_name not in reception_products
            ]

            specs["reception"].extend(reception_products)
            specs["storage"].extend(storage_products)
            remaining_reception_capacity -= reception_count

        if specs["reception"]:
            broken_count = random.randint(0, min(3, len(specs["reception"])))
            specs["broken"] = set(random.sample(specs["reception"], k=broken_count))

        return specs


    def _place_episode_products(self, batch_size: int):
        products_by_name = {obj.name: obj for obj in self.product_objects}

        reception_cells = self.object_slots["reception_returns"].copy()

        for product_name in self.episode_product_specs["reception"]:
            obj = products_by_name[product_name]
            state = BROKEN_STATE if product_name in self.episode_product_specs["broken"] else CLEAN_STATE
            cell_idx = random.randrange(len(reception_cells))
            cell = reception_cells.pop(cell_idx)
            self._place_product(obj, cell, state, batch_size)

        storage_cell_cursors = {product_type: 0 for product_type in PRODUCT_TYPES}

        for product_name in self.episode_product_specs["storage"]:
            obj = products_by_name[product_name]
            product_type = self._product_type_from_name(product_name)
            table_name = CATEGORY_STORAGE_TABLE[product_type]

            grids = self.object_slots[f"{table_name}_products"]
            cell_idx = storage_cell_cursors[product_type]
            cell = grids[product_type][cell_idx]
            storage_cell_cursors[product_type] += 1

            self._place_product(obj, cell, CLEAN_STATE, batch_size)

    def _product_type_from_name(self, name: str):
        for product_type in PRODUCT_TYPES:
            if name == product_type or name.startswith(f"{product_type}_"):
                return product_type
        raise ValueError(f"Unknown product type: {name}")

    def _place_product(self, obj, xyz, state: int, batch_size: int):
        obj.set_pose(self._pose_from_xyz(xyz, batch_size))
        self.object_states[obj.name] = torch.full(
            (self.num_envs,),
            state,
            dtype=torch.int32,
            device=self.device,
        )
        self.objects.append(obj)

    def _pose_from_xyz(self, xyz, batch_size: int):
        p = torch.tensor(xyz, device=self.device, dtype=torch.float32).repeat(batch_size, 1)
        return Pose.create_from_pq(p=p, q=[1, 0, 0, 0])

    def _reset_robot(self, agent_idx: int, slot_name: str):
        self.agent.agents[agent_idx].reset(self.panda_qpos)
        self.agent.agents[agent_idx].robot.set_pose(self.robot_slots[slot_name])

    @property
    def reception_robot(self) -> Panda:
        return self.agent.agents[0]

    @property
    def preparation_robot_a(self) -> Panda:
        return self.agent.agents[1]

    @property
    def preparation_robot_b(self) -> Panda:
        return self.agent.agents[2]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-5, -6, 1.5], [0.1, 0.5, -1])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=768,
            height=768,
            fov=1.0,
            near=0.01,
            far=100,
        )

    def get_magma_extra_state(self):
        out = {}

        if hasattr(self, "object_states"):
            out["object_states"] = {
                name: state.clone()
                for name, state in self.object_states.items()
            }

        if hasattr(self, "package_states"):
            out["package_states"] = {
                name: state.clone()
                for name, state in self.package_states.items()
            }

        return out

    def set_magma_extra_state(self, state: dict):
        if "object_states" in state and hasattr(self, "object_states"):
            for name, value in state["object_states"].items():
                if name in self.object_states:
                    self.object_states[name] = value.clone().to(device=self.device)

        if "package_states" in state and hasattr(self, "package_states"):
            for name, value in state["package_states"].items():
                if name in self.package_states:
                    self.package_states[name] = value.clone().to(device=self.device)

    def _static_state(self):
        return torch.full((self.num_envs,), STATIC_STATE, dtype=torch.int32, device=self.device)

    def _get_obs_extra(self, info: dict):
        obs = {
            "reception_tcp": {
                "pose": self.reception_robot.tcp.pose.raw_pose,
                "state": self._static_state(),
            },
            "preparation_a_tcp": {
                "pose": self.preparation_robot_a.tcp.pose.raw_pose,
                "state": self._static_state(),
            },
            "preparation_b_tcp": {
                "pose": self.preparation_robot_b.tcp.pose.raw_pose,
                "state": self._static_state(),
            },
            "trashcan": {
                "pose": self.trash_collision.pose.raw_pose,
                "state": self._static_state(),
            },
            "reception_helper_tray": {
                "pose": self.reception_helper_tray.pose.raw_pose,
                "state": self.package_states["reception_helper_tray"],
            },
        }

        for package in self.packages:
            obs[package.name] = {
                "pose": package.pose.raw_pose,
                "state": self.package_states[package.name],
            }

        for name, pose in self.robot_slots.items():
            obs[f"slot_{name}"] = {
                "pose": torch.tensor(
                    [*pose.p, *pose.q],
                    device=self.device,
                    dtype=torch.float32,
                ).repeat(self.num_envs, 1),
                "state": self._static_state(),
            }

        for obj in self.objects:
            obs[obj.name] = {
                "pose": obj.pose.raw_pose,
                "state": self.object_states[obj.name],
            }

        for i, xyz in enumerate(self.object_slots["output_packages"]):
            obs[f"output_package_slot_{i}"] = {
                "pose": torch.tensor(
                    [xyz[0], xyz[1], xyz[2], 1, 0, 0, 0],
                    device=self.device,
                    dtype=torch.float32,
                ).repeat(self.num_envs, 1),
                "state": self._static_state(),
            }

        reception_cells = self.object_slots["reception_returns"]
        reception_grid_center = torch.tensor(
            [
                sum(cell[0] for cell in reception_cells) / len(reception_cells),
                sum(cell[1] for cell in reception_cells) / len(reception_cells),
                self.cube_half_size,
                1,
                0,
                0,
                0,
            ],
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.num_envs, 1)

        obs["reception_grid"] = {
            "pose": reception_grid_center,
            "state": self._static_state(),
        }

        for table_key in ("storage_a_products", "storage_b_products"):
            for product_type, cells in self.object_slots[table_key].items():
                center = torch.tensor(
                    [
                        sum(cell[0] for cell in cells) / len(cells),
                        sum(cell[1] for cell in cells) / len(cells),
                        self.cube_half_size,
                        1,
                        0,
                        0,
                        0,
                    ],
                    device=self.device,
                    dtype=torch.float32,
                ).repeat(self.num_envs, 1)

                obs[f"{product_type}_grid"] = {
                    "pose": center,
                    "state": self._static_state(),
                }

        return obs

    def move_robot_to_slot(self, agent_idx: int, slot_name: str, env_id: int) -> str:
        if agent_idx < 0 or agent_idx >= len(self.agent.agents):
            return f"Invalid agent_idx: {agent_idx}"

        if slot_name not in self.robot_slots:
            return f"Invalid slot_name: {slot_name}"

        if not self._is_slot_allowed_for_robot(agent_idx, slot_name):
            return f"Robot {agent_idx} cannot access slot {slot_name}"

        for other_idx, occupied_slot in enumerate(self.current_robot_slots):
            if other_idx != agent_idx and occupied_slot == slot_name:
                return "Already occupied"

        robot = self.agent.agents[agent_idx].robot
        raw_pose = robot.get_pose().raw_pose.clone()
        slot_pose = self.robot_slots[slot_name]

        raw_pose[env_id, 0:3] = torch.tensor(
            slot_pose.p,
            device=raw_pose.device,
            dtype=raw_pose.dtype,
        )
        raw_pose[env_id, 3:7] = torch.tensor(
            slot_pose.q,
            device=raw_pose.device,
            dtype=raw_pose.dtype,
        )

        robot.set_pose(Pose(raw_pose=raw_pose))
        self.current_robot_slots[agent_idx] = slot_name
        return ""

    def _is_slot_allowed_for_robot(self, agent_idx: int, slot_name: str):
        if agent_idx == 0:
            return (
                slot_name in {"reception", "trash"}
                or slot_name.startswith("storage_")
            )
        return (
            slot_name.startswith("output_")
            or slot_name.startswith("storage_")
        )

    def move_preparation_robot_to_storage_zone(self, agent_idx: int, slot_name: str, env_id: int) -> str:
        if not slot_name.startswith("storage_"):
            return f"Invalid storage slot: {slot_name}"
        return self.move_robot_to_slot(agent_idx, slot_name, env_id)

from typing import Dict, Optional
import torch
from magma_core.simulation.data_structures import (
    EnvStateUpdate,
    Observation,
    ToolExecution,
    ToolResult,
    Log,
    Trajectory,
)
from magma_core.simulation.tools import BaseToolsAPI, register_tool
from magma_core.simulation.utils.env_utils import is_object_inside_target
from .attributes import PRODUCT_TYPES, TRASHCAN_PLACEMENT_THRESHOLD
import numpy as np
import sapien
from magma_core.simulation.utils.gripper_utils import find_object_in_gripper, is_object_in_gripper
from magma_scenarios.utils import compute_grasp_trajectory, compute_drop_trajectory
from magma_scenarios.templates.errors import OneShotToolFailureError
from magma_scenarios.templates.tools import PlacementGrid

PACKAGE_FREE = 0
PACKAGE_RECEPTION_ROBOT = 1
PACKAGE_PREPARATION_A = 2
PACKAGE_PREPARATION_B = 3

BROKEN = 0
CLEAN = 1

ROBOT_PACKAGE_STATES = {
    "reception_robot": PACKAGE_RECEPTION_ROBOT,
    "preparation_robot_a": PACKAGE_PREPARATION_A,
    "preparation_robot_b": PACKAGE_PREPARATION_B,
}

OUTPUT_TARGET_TO_SLOT = {
    "priority_bay_0": "output_0",
    "priority_bay_1": "output_1",
    "standard_bay_0": "output_2",
    "standard_bay_1": "output_3",
}

class AdvancedDeliveryTools(BaseToolsAPI):
    
    PACKAGE_NAMES = [
    "priority_0",
    "priority_1",
    "standard_0",
    "standard_1",
    ]
    HELPER_TRAY_NAME = "reception_helper_tray"
    PACKAGE_OCCUPIED_DISTANCE = 0.25
    SLOT_OCCUPIED_DISTANCE = 0.8
    PACKAGE_CONTENT_DISTANCE = 0.22

    OUTPUT_SLOT_TO_BAY = {
        0: "priority_bay_0",
        1: "priority_bay_1",
        2: "standard_bay_0",
        3: "standard_bay_1",
    }

    MOVE_TARGET_TO_SLOT = {
        "reception": "reception",
        "electronics_spot": "storage_a_left_0",
        "drinks_spot": "storage_a_right_1",
        "snacks_spot": "storage_a_left_2",
        "hygiene_spot": "storage_b_right_3",
        "textile_spot": "storage_b_left_4",
        "priority_bay_0": "output_0",
        "priority_bay_1": "output_1",
        "standard_bay_0": "output_2",
        "standard_bay_1": "output_3",
    }

    PRODUCT_TYPE_TO_SLOT = {
        "electronics": "storage_a_left_0",
        "drinks": "storage_a_right_1",
        "snacks": "storage_a_left_2",
        "hygiene": "storage_b_right_3",
        "textile": "storage_b_left_4",
    }

    reception_placement_grid = PlacementGrid(
        name="advanced_delivery_reception",
        rows=3,
        columns=3,
        cell_spacing=0.08,
        selection_order=[1, 4, 7, 5, 3, 2, 8, 0, 6],
    )
    storage_placement_grid = PlacementGrid(
        name="advanced_delivery_storage",
        rows=2,
        columns=2,
        cell_spacing=0.16,
        selection_order=[1, 3, 0, 2],
    )


    def _robot_base_pos(self, robot, env_id: int):
        return robot.robot.get_pose().raw_pose[env_id, :3]

    def _tcp_key(self, robot_name: str):
        return {
            "reception_robot": "reception_tcp",
            "preparation_robot_a": "preparation_a_tcp",
            "preparation_robot_b": "preparation_b_tcp",
        }[robot_name]

    def _product_type_from_name(self, name: str) -> Optional[str]:
        for product_type in PRODUCT_TYPES:
            if name == product_type or name.startswith(f"{product_type}_") and not name.endswith("grid"):
                return product_type
        return None

    def _product_poses(self, obs: Observation, env_id: int):
        # return {name : pose} for all existing product
        extra = obs.maniskill_obs["extra"]
        return {
            name: entry["pose"][env_id][:3]
            for name, entry in extra.items()
            if self._product_type_from_name(name) is not None
        }

    def _tcp_pose(self, obs: Observation, env_id: int):
        return obs.maniskill_obs["extra"][self._tcp_key(obs.selected_robot_name)]["pose"][env_id][:3]

    def _held_object(self, obs: Observation, env_id: int):
        return find_object_in_gripper(self._tcp_pose(obs, env_id), self._product_poses(obs, env_id))

    def _slot_pose(self, obs: Observation, env_id: int, slot_name: str):
        extra = obs.maniskill_obs["extra"]
        obs_slot_name = f"slot_{slot_name}"

        if obs_slot_name not in extra:
            return None

        return extra[obs_slot_name]["pose"][env_id]

    def _allocate_product_cell(
        self,
        obs: Observation,
        env_id: int,
        support_name: str,
        product_name: str,
        grid: PlacementGrid,
    ):
        support_pose = obs.maniskill_obs["extra"][support_name]["pose"][env_id]
        cell = grid.allocate(
            center_pose=support_pose,
            object_poses=self._product_poses(obs, env_id),
            batch_context=obs.tool_batch_context,
            owner=product_name,
            reservation_namespace=f"advanced_delivery:{support_name}",
            excluded_objects=[product_name],
        )
        return None if cell is None else cell.world_position

    def _table_target_position(self, obs: Observation, env_id: int, product_name: str):
        extra = obs.maniskill_obs["extra"]
        robot = self.get_agent(obs.selected_robot_name)
        robot_pos = self._robot_base_pos(robot, env_id)

        # If the robot is at the reception slot, "table" means reception table.
        reception_slot = self._slot_pose(obs, env_id, "reception")

        robot_is_at_reception = (
            reception_slot is not None
            and torch.norm(robot_pos[:2] - reception_slot[:2]
            ).item() < self.SLOT_OCCUPIED_DISTANCE
        )

        if robot_is_at_reception:
            if "reception_grid" not in extra:
                return None

            return self._allocate_product_cell(
                obs,
                env_id,
                support_name="reception_grid",
                product_name=product_name,
                grid=self.reception_placement_grid,
            )

        # Otherwise, "table" means the storage grid matching the product type.
        product_type = self._product_type_from_name(product_name)
        if product_type is None:
            return None

        grid_name = f"{product_type}_grid"
        if grid_name not in extra:
            return None

        return self._allocate_product_cell(
            obs,
            env_id,
            support_name=grid_name,
            product_name=product_name,
            grid=self.storage_placement_grid,
        )

    
    def _package_in_obs(self, extra: Dict):
        names = [
            name
            for name in self.PACKAGE_NAMES
            if name in extra
        ]

        if self.HELPER_TRAY_NAME in extra:
            names.append(self.HELPER_TRAY_NAME)

        return names

    def _held_package(self, obs: Observation, env_id: int):
        extra = obs.maniskill_obs["extra"]
        robot_state = ROBOT_PACKAGE_STATES[obs.selected_robot_name]

        for package_name in self._package_in_obs(extra):
            state = int(extra[package_name]["state"][env_id].item())

            if state == robot_state:
                return package_name

        return None

    def _package_target_position(
        self,
        obs: Observation,
        env_id: int,
        product_name: str,
    ):
        package_name = self._held_package(obs, env_id)
        if package_name is None:
            return None, None

        return package_name, self._allocate_product_cell(
            obs,
            env_id,
            support_name=package_name,
            product_name=product_name,
            grid=self.storage_placement_grid,
        )


    def _objects_on_package(self, obs: Observation, env_id: int, package_name: str):
        extra = obs.maniskill_obs["extra"]
        package_pos = extra[package_name]["pose"][env_id][:3]
        objects = []

        for name, entry in extra.items():
            if name in self._package_in_obs(extra):
                continue
            if name.startswith("slot_"):
                continue
            if name.endswith("_tcp"):
                continue
            if name in {"trashcan"}:
                continue

            product_type = self._product_type_from_name(name)
            if product_type is None:
                continue

            obj_pos = entry["pose"][env_id][:3]
            if torch.norm(obj_pos[:2] - package_pos[:2]).item() < self.PACKAGE_CONTENT_DISTANCE:
                objects.append(name)

        return objects


    def _slot_is_occupied(self, obs: Observation, env_id: int, slot_name: str) -> Optional[str]:
        extra = obs.maniskill_obs["extra"]
        slot_obs_name = f"slot_{slot_name}"

        if slot_obs_name not in extra:
            return None

        slot_pos = extra[slot_obs_name]["pose"][env_id][:3]

        for name in ("reception_tcp", "preparation_a_tcp", "preparation_b_tcp"):
            if name not in extra:
                continue
            tcp_pos = extra[name]["pose"][env_id][:3]
            if torch.norm(tcp_pos[:2] - slot_pos[:2]).item() < self.SLOT_OCCUPIED_DISTANCE:
                return name

        return None

    def _robot_is_at_slot(self, obs: Observation, env_id: int, robot_name: str, slot_name: str):
        extra = obs.maniskill_obs["extra"]
        slot_obs_name = f"slot_{slot_name}"
        tcp_name = self._tcp_key(robot_name)

        if slot_obs_name not in extra or tcp_name not in extra:
            return False

        slot_pos = extra[slot_obs_name]["pose"][env_id][:3]
        tcp_pos = extra[tcp_name]["pose"][env_id]

        return torch.norm(tcp_pos[:2] - slot_pos[:2]).item() < self.SLOT_OCCUPIED_DISTANCE

    def _right_offset_from_quat(self, q, distance: float, like_tensor):
        qw, qx, qy, qz = q

        yaw = torch.atan2(
            2 * (qw * qz + qx * qy),
            1 - 2 * (qy * qy + qz * qz),
        )

        right = torch.stack([
            torch.sin(yaw),
            -torch.cos(yaw),
            torch.zeros((), device=like_tensor.device, dtype=like_tensor.dtype),
        ])

        return right * distance

    def _package_product_counts(self, obs: Observation, env_id: int, package_name: str) -> tuple[Dict[str, int], list[str]]:
        extra = obs.maniskill_obs["extra"]

        product_counts = {
            product_type: 0
            for product_type in PRODUCT_TYPES
        }

        broken_objects = []

        for object_name in self._objects_on_package(obs, env_id, package_name):
            product_type = self._product_type_from_name(object_name)

            if product_type is None:
                continue

            product_counts[product_type] += 1

            object_state = int(
                extra[object_name]["state"][env_id].item()
            )

            if object_state == BROKEN:
                broken_objects.append(object_name)

        return product_counts, broken_objects

    
    @register_tool(
        description="Move the selected robot to a destination.",
        params_spec={"target": {"description": "Target destination.", "type": str}},
        errors=[OneShotToolFailureError],
    )
    def move_to(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]
        target = params["target"]

        selected_robot_name = obs.selected_robot_name
        robot = self.get_agent(selected_robot_name)

        carried_package = self._held_package(obs, env_id)

        if target not in self.MOVE_TARGET_TO_SLOT:
            return ToolExecution([], verifier=None, reason=f"Unknown target {target}.")
        slot_name = self.MOVE_TARGET_TO_SLOT[target]

        if selected_robot_name == "reception_robot":
            slot_is_accessible = slot_name == "reception" or slot_name.startswith("storage_")
        else:
            slot_is_accessible = slot_name.startswith(("storage_", "output_"))

        if not slot_is_accessible:
            return ToolExecution(
                [],
                verifier=None,
                reason=f"{target} is not accessible to {selected_robot_name}.",
            )

        if self._robot_is_at_slot(obs, env_id, selected_robot_name, slot_name):
            def verifier(new_obs: Dict) -> ToolResult:
                return ToolResult(
                    True,
                    reason=f"{selected_robot_name} is at {target}.",
                )

            return ToolExecution(poses=["OK"], verifier=verifier)

        robot_in_target = self._slot_is_occupied(obs, env_id, slot_name)
        if robot_in_target is not None:
            return ToolExecution([], verifier=None, reason=f"{target} is already occupied by {robot_in_target}.")

        slot_pose = self._slot_pose(obs, env_id, slot_name)
        if slot_pose is None:
            return ToolExecution([], verifier=None, reason=f"Unknown {target}.")

        target_robot_xyz = slot_pose[:3]
        target_robot_q = slot_pose[3:7]

        target_robot_state = robot.robot.get_state()[env_id].clone()
        target_robot_state[0:3] = target_robot_xyz
        target_robot_state[3:7] = target_robot_q
        target_robot_state[7:13] = 0

        state_updates = [
            EnvStateUpdate(path=("articulations", robot.robot.name), value=target_robot_state)
        ]

        if carried_package is not None:
            old_package_pose = extra[carried_package]["pose"][env_id].clone()

            target_package_xyz = target_robot_xyz.clone()
            target_package_xyz -= self._right_offset_from_quat(
                target_robot_q,
                distance=0.55,
                like_tensor=target_robot_xyz,
            )
            target_package_xyz[2] = 0.02

            package_delta = target_package_xyz - old_package_pose[:3]

            target_package_state = torch.zeros(
                13,
                device=old_package_pose.device,
                dtype=old_package_pose.dtype,
            )
            target_package_state[0:3] = target_package_xyz
            target_package_state[3:7] = target_robot_q
            target_package_state[7:13] = 0

            state_updates.append(EnvStateUpdate(path=("actors", carried_package), value=target_package_state))

            for object_name in self._objects_on_package(obs, env_id, carried_package):
                old_obj_pose = extra[object_name]["pose"][env_id].clone()

                target_obj_state = torch.zeros(
                    13,
                    device=old_obj_pose.device,
                    dtype=old_obj_pose.dtype,
                )
                target_obj_state[0:3] = old_obj_pose[:3] + package_delta
                target_obj_state[3:7] = old_obj_pose[3:7]
                target_obj_state[7:13] = 0

                state_updates.append(EnvStateUpdate(path=("actors", object_name), value=target_obj_state))

        def verifier(new_obs: Dict) -> ToolResult:
            return ToolResult(
                True,
                reason=f"{selected_robot_name} is at {target}.",
                state_updates=state_updates,
            )

        return ToolExecution(poses=["OK"], verifier=verifier)

    @register_tool(
        description="Take an available package tray from the output table.",
        params_spec={},
        errors=[OneShotToolFailureError],
    )
    def take_package(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]
        selected_robot_name = obs.selected_robot_name

        if selected_robot_name == "reception_robot":
            return ToolExecution([], verifier=None, reason="The reception robot already has its fixed helper tray.")

        robot_state = ROBOT_PACKAGE_STATES[selected_robot_name]

        if self._held_package(obs, env_id) is not None:
            return ToolExecution([], verifier=None, reason=f"{selected_robot_name} already holds a package.")

        current_output_idx = None

        robot = self.get_agent(selected_robot_name)
        robot_pos = self._robot_base_pos(robot, env_id)

        for i in range(4):
            slot_name = f"output_{i}"
            slot_pose = self._slot_pose(obs, env_id, slot_name)

            if slot_pose is None:
                continue

            if torch.norm(robot_pos[:2] - slot_pose[:3][:2]).item() < self.SLOT_OCCUPIED_DISTANCE:
                current_output_idx = i
                break

        if current_output_idx is None:
            return ToolExecution([], verifier=None, reason=f"{selected_robot_name} is not in front of an output buy.")

        package_name = self.PACKAGE_NAMES[current_output_idx]

        if package_name not in extra:
            return ToolExecution([], verifier=None, reason=f"{package_name} does not exist.")

        package_state = int(extra[package_name]["state"][env_id].item())

        if package_state != PACKAGE_FREE:
            return ToolExecution([], verifier=None, reason=f"{package_name} is not free.")

        old_package_pose = extra[package_name]["pose"][env_id].clone()

        target_package_xyz = robot_pos.clone()
        robot_q = robot.robot.get_pose().raw_pose[env_id, 3:7]

        target_package_xyz -= self._right_offset_from_quat(
            robot_q,
            distance=0.55,
            like_tensor=robot_pos,
        )
        target_package_xyz[2] = 0.02

        target_package_state = torch.zeros(
            13,
            device=old_package_pose.device,
            dtype=old_package_pose.dtype,
        )
        target_package_state[0:3] = target_package_xyz
        target_package_state[3:7] = old_package_pose[3:7]
        target_package_state[7:13] = 0

        state_updates = [EnvStateUpdate(path=("actors", package_name), value=target_package_state),
            EnvStateUpdate(
                path=("magma_extra_state", "package_states", package_name),
                value=torch.tensor(robot_state, device=old_package_pose.device, dtype=torch.int32))
        ]

        def verifier(new_obs: Dict) -> ToolResult:
            return ToolResult(True, reason=f"{selected_robot_name} took {package_name}.", state_updates=state_updates)

        return ToolExecution(poses=["OK"], verifier=verifier)

    @register_tool(
        description="Put the held package tray on the output table.",
        params_spec={},
        errors=[OneShotToolFailureError],
    )
    def drop_package(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]
        selected_robot_name = obs.selected_robot_name

        if selected_robot_name == "reception_robot":
            return ToolExecution([], verifier=None, reason="The reception robot cannot drop its fixed helper package.")

        carried_package = self._held_package(obs, env_id)

        if carried_package is None:
            return ToolExecution([], verifier=None, reason=f"{selected_robot_name} does not hold a package.")

        robot = self.get_agent(selected_robot_name)
        robot_pos = self._robot_base_pos(robot, env_id)

        output_idx = None
        for i in range(4):
            slot_pose = self._slot_pose(obs, env_id, f"output_{i}")
            if slot_pose is None:
                continue

            if torch.norm(robot_pos[:2] - slot_pose[:2]).item() < self.SLOT_OCCUPIED_DISTANCE:
                output_idx = i
                break

        if output_idx is None:
            return ToolExecution([], verifier=None, reason=f"{selected_robot_name} is not in front of an output slot.")

        package_slot_name = f"output_package_slot_{output_idx}"

        if package_slot_name not in extra:
            return ToolExecution([], verifier=None, reason=f"{package_slot_name} is not available in observation.",)

        # This output place must not already contain a free package.
        for package_name in self.PACKAGE_NAMES:
            if package_name not in extra or package_name == carried_package:
                continue

            state = int(extra[package_name]["state"][env_id].item())
            if state != PACKAGE_FREE:
                continue

            package_pos = extra[package_name]["pose"][env_id][:3]
            target_pos = extra[package_slot_name]["pose"][env_id][:3]

            if torch.norm(package_pos[:2] - target_pos[:2]).item() < self.PACKAGE_OCCUPIED_DISTANCE:
                return ToolExecution([], verifier=None, reason=f"Output package slot {output_idx} already contains {package_name}.")

        old_package_pose = extra[carried_package]["pose"][env_id].clone()
        target_slot_pose = extra[package_slot_name]["pose"][env_id]

        target_package_xyz = target_slot_pose[:3].clone()
        target_package_xyz[2] = target_slot_pose[2]

        package_delta = target_package_xyz - old_package_pose[:3]

        target_package_state = torch.zeros(
            13,
            device=old_package_pose.device,
            dtype=old_package_pose.dtype,
        )
        target_package_state[0:3] = target_package_xyz
        target_package_state[3:7] = old_package_pose[3:7]
        target_package_state[7:13] = 0

        state_updates = [
            EnvStateUpdate(path=("actors", carried_package), value=target_package_state),
            EnvStateUpdate(
                path=("magma_extra_state", "package_states", carried_package),
                value=torch.tensor(PACKAGE_FREE, device=old_package_pose.device, dtype=torch.int32)
            )
        ]

        for object_name in self._objects_on_package(obs, env_id, carried_package):
            old_obj_pose = extra[object_name]["pose"][env_id].clone()

            target_obj_state = torch.zeros(
                13,
                device=old_obj_pose.device,
                dtype=old_obj_pose.dtype,
            )
            target_obj_state[0:3] = old_obj_pose[:3] + package_delta
            target_obj_state[3:7] = old_obj_pose[3:7]
            target_obj_state[7:13] = 0

            state_updates.append(EnvStateUpdate(path=("actors", object_name), value=target_obj_state))

        def verifier(new_obs: Dict) -> ToolResult:
            return ToolResult(
                True,
                reason=f"{selected_robot_name} dropped {carried_package} on output position {output_idx}.",
                state_updates=state_updates,
            )

        return ToolExecution(poses=["OK"], verifier=verifier)


    @register_tool(
        description="Take a product.",
        params_spec={"name": {"description": "Product name, for example electronics_0.", "type": str}},
        errors=[OneShotToolFailureError],
    )
    def pick(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]
        name = params["name"]
        robot_name = obs.selected_robot_name

        if name not in extra:
            return ToolExecution([], verifier=None, reason=f"{name} was not detected.")

        if self._product_type_from_name(name) is None:
            return ToolExecution([], verifier=None, reason=f"{name} is not a product.")

        if self._held_object(obs, env_id) is not None:
            return ToolExecution([], verifier=None, reason="The gripper is not empty.")

        object_is_in_robot_zone = False
        held_package = self._held_package(obs, env_id)

        if held_package is not None:
            object_is_in_robot_zone = name in self._objects_on_package(
                obs,
                env_id,
                held_package,
            )

        if not object_is_in_robot_zone:
            object_position = extra[name]["pose"][env_id][:3]
            reception_grid = extra.get("reception_grid")

            if reception_grid is not None and is_object_inside_target(
                object_position,
                reception_grid["pose"][env_id],
                thresh=self.TABLE_DETECTION_DISTANCE,
            ):
                object_is_in_robot_zone = self._robot_is_at_slot(
                    obs,
                    env_id,
                    robot_name,
                    "reception",
                )
            else:
                product_type = self._product_type_from_name(name)
                grid_name = f"{product_type}_grid"
                product_grid = extra.get(grid_name)

                if product_grid is not None and is_object_inside_target(
                    object_position,
                    product_grid["pose"][env_id],
                    thresh=self.TABLE_DETECTION_DISTANCE,
                ):
                    object_is_in_robot_zone = self._robot_is_at_slot(
                        obs,
                        env_id,
                        robot_name,
                        self.PRODUCT_TYPE_TO_SLOT[product_type],
                    )

        if not object_is_in_robot_zone:
            return ToolExecution(
                [],
                verifier=None,
                reason=f"Object {name} is not in the same zone as robot {robot_name}.",
            )

        obj_pose = extra[name]["pose"][env_id]

        poses = compute_grasp_trajectory(
            self.get_agent(robot_name),
            obj_pose.cpu().numpy(),
            add_seuil=0.05,
        )

        def verifier(new_obs: dict) -> ToolResult:
            new_extra = new_obs["extra"]

            if not is_object_in_gripper(
                new_extra[self._tcp_key(robot_name)]["pose"][env_id],
                new_extra[name]["pose"][env_id],
                threshold=0.05,
            ):
                return ToolResult(False, reason=f"Failed to pick {name}.")

            return ToolResult(
                True,
                reason=f"{name} is in the gripper.",
                logs=Log(content={"object": name, "robot": robot_name}),
            )

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            context={"target_name": name},
            allowed_moving_actors=[name],
        )

    @register_tool(
        description="Put the held product on a table, tray, trashcan, or package.",
        params_spec={"target": {"description": "Target: table, tray, trashcan, or package.", "type": str}},
        errors=[OneShotToolFailureError],
    )
    def put(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]
        target = params["target"]
        robot_name = obs.selected_robot_name

        if target == "trashcan" and robot_name != "reception_robot":
            return ToolExecution(
                [],
                verifier=None,
                reason="Only the reception robot can put products in the trashcan.",
            )

        held_object = self._held_object(obs, env_id)
        if held_object is None:
            return ToolExecution([], verifier=None, reason="The gripper is empty.")

        object_state = int(extra[held_object]["state"][env_id].item())
        
        resolved_target = target

        if target == "table":
            product_type = self._product_type_from_name(held_object)

            if self._robot_is_at_slot(obs, env_id, robot_name, "reception"):
                resolved_target = "reception_grid"
            elif product_type is not None and self._robot_is_at_slot(
                obs,
                env_id,
                robot_name,
                self.PRODUCT_TYPE_TO_SLOT[product_type],
            ):
                resolved_target = f"{product_type}_grid"
            else:
                return ToolExecution(
                    [],
                    verifier=None,
                    reason=f"Target table is not in the same zone as robot {robot_name}.",
                )

            target_position = self._table_target_position(obs, env_id, held_object)

        elif target == "trashcan":
            if not self._robot_is_at_slot(obs, env_id, robot_name, "reception"):
                return ToolExecution(
                    [],
                    verifier=None,
                    reason=f"Target trashcan is not in the same zone as robot {robot_name}.",
                )

            if target not in extra:
                return ToolExecution([], verifier=None, reason="trashcan was not detected.")

            trash_pose = extra["trashcan"]["pose"][env_id]
            target_position = trash_pose[:3].cpu().numpy()
            target_position[2] += 0.08

        elif target in {"package", "tray"}:
            package_name, target_position = self._package_target_position(
                obs,
                env_id,
                held_object,
            )
            if package_name is None:
                return ToolExecution([], verifier=None, reason="The robot does not hold a package.")

        else:
            return ToolExecution([], verifier=None, reason=f"Unknown target: {target}.")

        if target_position is None:
            return ToolExecution([], verifier=None, reason=f"There is no free cell in {target}.")

        tcp_pose = self._tcp_pose(obs, env_id)

        poses: Trajectory = [
            sapien.Pose(
                p=tcp_pose.cpu().numpy() + np.array([0, 0, 0.1]),
                q=(0, 1, 0, 0),
            )
        ]
        poses.extend(
            compute_drop_trajectory(
                self.get_agent(robot_name),
                drop_pose=target_position,
                drop_seuil=0.03,
                approach_seuil=0.13,
            )
        )

        def verifier(new_obs: dict) -> ToolResult:
            new_extra = new_obs["extra"]
            obj_pose = new_extra[held_object]["pose"][env_id]

            if is_object_in_gripper(new_extra[self._tcp_key(robot_name)]["pose"][env_id], obj_pose):
                return ToolResult(False, reason=f"{held_object} is still in the gripper.")

            target_tensor = torch.tensor(
                target_position,
                device=obj_pose.device,
                dtype=obj_pose.dtype,
            )
            if target == "trashcan":
                placement_distance = torch.norm(
                    obj_pose[:2] - target_tensor[:2]
                ).item()
                placement_threshold = TRASHCAN_PLACEMENT_THRESHOLD
            else:
                placement_distance = torch.norm(
                    obj_pose[:3] - target_tensor[:3]
                ).item()
                placement_threshold = 0.10

            if placement_distance > placement_threshold:
                return ToolResult(False, reason=f"{held_object} is not in {target}.")

            return ToolResult(
                True,
                reason=f"Successfully placed {held_object} on {target}.",
                logs=Log(
                    content={
                        "object": held_object,
                        "target": target,
                        "resolved_target": resolved_target,
                        "state": object_state,
                        "robot": robot_name,
                    }
                ),
            )

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            allowed_moving_actors=[held_object],
        )

    @register_tool(
        description="Inspect the held product.",
        params_spec={},
        is_detection=True,
        errors=[OneShotToolFailureError],
    )
    def inspect(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]
        robot_name = obs.selected_robot_name

        held_object = self._held_object(obs, env_id)
        if held_object is None:
            return ToolExecution([], verifier=None, reason="The gripper is empty.")

        if held_object not in extra:
            return ToolExecution([], verifier=None, reason=f"{held_object} was not detected.")

        state = int(extra[held_object]["state"][env_id].item())

        if state == BROKEN:
            condition = "broken"
        elif state == CLEAN:
            condition = "clean"
        else:
            return ToolExecution([], verifier=None, reason=f"{held_object} has an unknown state: {state}.")

        def verifier(new_obs: dict) -> ToolResult:
            return ToolResult(
                True,
                reason=f"{robot_name} inspected {held_object}: it is {condition}.",
                logs=Log(
                    content={"object": held_object, "state": condition, "robot": robot_name}
                ),
            )

        return ToolExecution(poses=["OK"], verifier=verifier)

    TABLE_DETECTION_DISTANCE = 0.35

    def _products_near_center(self, obs: Observation, env_id: int, center, distance: float):
        products = []

        for product_name, product_pos in self._product_poses(obs, env_id).items():
            if torch.norm(product_pos[:2] - center[:2]).item() < distance:
                products.append(product_name)

        return sorted(products)

    def _current_table_products(self, obs: Observation, env_id: int):
        extra = obs.maniskill_obs["extra"]
        robot = self.get_agent(obs.selected_robot_name)
        robot_pos = self._robot_base_pos(robot, env_id)

        candidates = []

        if "reception_grid" in extra:
            candidates.append(("reception_table", extra["reception_grid"]["pose"][env_id][:3]))

        for product_type in PRODUCT_TYPES:
            grid_name = f"{product_type}_grid"
            if grid_name in extra:
                candidates.append((f"{product_type}_table", extra[grid_name]["pose"][env_id][:3]))

        if not candidates:
            return None, []

        closest_name = None
        closest_center = None
        closest_distance = float("inf")

        for table_name, center in candidates:
            distance = torch.norm(robot_pos[:2] - center[:2]).item()
            if distance < closest_distance:
                closest_name = table_name
                closest_center = center
                closest_distance = distance

        if closest_center is None:
            return None, []

        products = self._products_near_center(
            obs,
            env_id,
            closest_center,
            self.TABLE_DETECTION_DISTANCE,
        )

        return closest_name, products

    @register_tool(
        description="List nearby products and their locations.",
        params_spec={},
        is_detection=True,
        errors=[OneShotToolFailureError],
    )
    def detection(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        selected_robot_name = obs.selected_robot_name

        table_name, table_products = self._current_table_products(obs, env_id)

        package_name = self._held_package(obs, env_id)
        package_products = []

        if package_name is not None:
            package_products = self._objects_on_package(obs, env_id, package_name)

        table_text = ", ".join(table_products) if table_products else "nothing"
        package_text = ", ".join(package_products) if package_products else "nothing"

        if table_name is None:
            table_name = "current table"

        def verifier(new_obs: Dict) -> ToolResult:
            return ToolResult(
                True,
                reason=(
                    f"On {table_name}, there is: {table_text}. "
                    f"In the package, there is: {package_text}."
                ),
                logs=Log(
                    content={
                        "robot": selected_robot_name,
                        "table": table_name,
                        "table_products": table_products,
                        "package": package_name,
                        "package_products": package_products,
                    }
                ),
            )

        return ToolExecution(poses=["OK"], verifier=verifier)


    def _output_idx_in_front_of_robot(self, obs: Observation, env_id: int):
        robot = self.get_agent(obs.selected_robot_name)
        robot_pos = self._robot_base_pos(robot, env_id)

        for i in range(4):
            slot_pose = self._slot_pose(obs, env_id, f"output_{i}")
            if slot_pose is None:
                continue

            if torch.norm(robot_pos[:2] - slot_pose[:2]).item() < self.SLOT_OCCUPIED_DISTANCE:
                return i

        return None

    def _package_at_output_idx(self, obs: Observation, env_id: int, output_idx: int):
        extra = obs.maniskill_obs["extra"]
        package_slot_name = f"output_package_slot_{output_idx}"

        if package_slot_name not in extra:
            return None

        target_pos = extra[package_slot_name]["pose"][env_id][:3]

        for package_name in self.PACKAGE_NAMES:
            if package_name not in extra:
                continue

            state = int(extra[package_name]["state"][env_id].item())
            if state != PACKAGE_FREE:
                continue

            package_pos = extra[package_name]["pose"][env_id][:3]
            if torch.norm(package_pos[:2] - target_pos[:2]).item() < self.PACKAGE_OCCUPIED_DISTANCE:
                return package_name

        return None

    @register_tool(
        description="Attach a shipping label to the package in front of the selected robot.",
        params_spec={
            "order_id": {"description": "Unique delivery reference.","type": str},
            "name": {"description": "Recipient name.","type": str},
            "city": {"description": "Destination city.", "type": str},
        },
        errors=[OneShotToolFailureError],
    )
    def attach_shipping_label(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        name = params["name"]
        city = params["city"]
        order_id = params["order_id"]

        output_idx = self._output_idx_in_front_of_robot(obs, env_id)
        if output_idx is None:
            return ToolExecution([], verifier=None, reason="The robot is not in front of an output bay.")

        package_name = self._package_at_output_idx(obs, env_id, output_idx)
        if package_name is None:
            return ToolExecution([], verifier=None, reason="There is no dropped package in front of the robot.")

        bay_name = self.OUTPUT_SLOT_TO_BAY[output_idx]

        product_counts, broken_objects = (
            self._package_product_counts(obs, env_id, package_name)
        )

        if not any(product_counts.values()):
            return ToolExecution(
                [],
                verifier=None,
                reason=(f"{package_name} is empty and cannot be labelled."),
            )

        if broken_objects:
            return ToolExecution(
                [],
                verifier=None,
                reason=(
                    f"{package_name} contains damaged products: {', '.join(broken_objects)}."
                ),
            )

        def verifier(new_obs: Dict) -> ToolResult:
            return ToolResult(
                True,
                reason=f"Shipping label attached to {package_name} for {name} in {city}.",
                logs=Log(
                    content={
                        "order_id": order_id,
                        "package": package_name,
                        "bay": bay_name,
                        "name": name,
                        "city": city,
                        "products": product_counts,
                    }
                ),
            )

        return ToolExecution(poses=["OK"], verifier=verifier)

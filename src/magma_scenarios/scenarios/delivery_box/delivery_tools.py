# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict
import torch

from magma_core.simulation.data_structures import (
    EnvStateUpdate,
    Log,
    Observation,
    ToolErrorSupport,
    ToolExecution,
    ToolResult,
)
from magma_core.simulation.tools import BaseToolsAPI, register_tool
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_core.simulation.utils.gripper_utils import find_object_in_gripper, is_object_in_gripper
from magma_scenarios.utils import compute_drop_trajectory, compute_grasp_trajectory
from magma_scenarios.templates.errors import OneShotToolFailureError
from magma_scenarios.templates.tools import PlacementGrid
from .attributes import PACKAGE_NAMES
PACKAGE_FREE = 0
PACKAGE_HELD = 1



HELD_PACKAGE_POSITIONS = {
    "packages_slot": torch.tensor([0.0, 0.6, 0.2]),
    "products_slot": torch.tensor([-0.7, -0.4, 0.2]),
}

PACKAGE_HOME_POSITIONS = {
    "package_1": torch.tensor([-0.60, 1.5, 0.0]),
    "package_2": torch.tensor([-0.20, 1.5, 0.0]),
    "package_3": torch.tensor([ 0.20, 1.5, 0.0]),
    "package_4": torch.tensor([ 0.60, 1.5, 0.0]),
}

class CycleTool(BaseToolsAPI):

    table_placement_grid = PlacementGrid(
        name="delivery_box_table",
        rows=3,
        columns=3,
        cell_spacing=0.15,
        selection_order=[0, 3, 6, 1, 4, 7, 2, 5, 8],
    )

    def _robot_is_at_slot(self, obs: Observation, env_id: int, slot_name: str) -> bool:
        robot_position = self.get_agent().robot.get_state()[env_id][:3]
        slot_position = obs.maniskill_obs["extra"][slot_name][env_id][:3]
        return bool(torch.linalg.vector_norm(robot_position - slot_position) < 0.05)

    def _held_package(self, obs: Observation, env_id: int):
        extra = obs.maniskill_obs["extra"]

        for name in PACKAGE_NAMES:
            if name not in extra:
                continue

            state = int(extra[name]["state"][env_id].item())
            if state == PACKAGE_HELD:
                return name

        return None


    def _objects_in_package(self, obs: Observation, env_id: int, package_name: str):
        extra = obs.maniskill_obs["extra"]
        package_pose = extra[package_name]["pose"][env_id]
        objects = []

        for product_type in obs.task_attributes["product_type"]:
            for name, entry in extra.items():
                if not name.startswith(f"{product_type}_"):
                    continue

                pose = entry["pose"] if isinstance(entry, dict) else entry

                if is_object_inside_target(pose[env_id], package_pose, keep_tensor=False):
                    objects.append(name)

        return objects


    def _package_state_at(self, extra: Dict, env_id: int, package_name: str, target_xyz: torch.Tensor):
        state = extra[package_name]["sim_state"][env_id].clone()
        state[:3] = target_xyz
        state[7:13] = 0
        return state

    def _package_product_counts(self, obs: Observation, env_id: int, package_name: str) -> Dict[str, int]:
        extra = obs.maniskill_obs["extra"]
        package_pose = extra[package_name]["pose"][env_id]

        counts = {product_type: 0 for product_type in obs.task_attributes["product_type"]}

        for product_type in counts:
            for object_name, entry in extra.items():
                if not object_name.startswith(f"{product_type}_"):
                    continue

                object_pose = (entry["pose"] if isinstance(entry, dict) else entry)

                if is_object_inside_target(object_pose[env_id], package_pose, keep_tensor=False):
                    counts[product_type] += 1

        return counts

    @register_tool(
        description="List the products in the package held by the robot.",
        params_spec={},
        is_detection=True,
    )
    def get_object_in_package(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        held_package = self._held_package(obs, env_id)

        if held_package is None:
            return ToolExecution(
                poses=[],
                verifier=None,
                reason="no package held by the robot",
            )

        package_objects = self._objects_in_package(obs, env_id, held_package)

        def verifier(new_obs: Dict) -> ToolResult:
            package_description = (
                f"{held_package} contains " + ", ".join(package_objects) + "."
                if package_objects
                else f"{held_package} contains no product."
            )

            return ToolResult(
                True,
                reason=package_description,
                context=package_objects,
                logs=Log(""),
            )

        return ToolExecution(poses=["OK"], verifier=verifier)

    @register_tool(
        description="Take an available product of a given type.",
        params_spec={
            "type_obj": {
                "type": str,
                "description": "The product type to take.",
            }
        },
        errors=[
            ToolErrorSupport(
                OneShotToolFailureError,
                pre=True,
                post=False,
            )
        ],
    )
    def take(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        if not self._robot_is_at_slot(obs, env_id, "products_slot"):
            return ToolExecution(
                poses=[],
                verifier=None,
                reason="The robot must be at products_slot to take a product.",
            )

        product_type = params["type_obj"]
        if product_type not in obs.task_attributes["product_type"]:
            return ToolExecution(
                poses=[],
                verifier=None,
                reason=(
                    f"Unknown product type {product_type}. Please use one of "
                    f"{obs.task_attributes['product_type']}."
                ),
            )

        extra = obs.maniskill_obs["extra"]
        product_types = obs.task_attributes["product_type"]

        reduced_obs = {
            name: value[env_id][:3]
            for name, value in extra.items()
            if (
                isinstance(value, torch.Tensor)
                and any(
                    name.startswith(f"{known_type}_")
                    for known_type in product_types
                )
            )
        }

        held_object = find_object_in_gripper(extra["agent_tcp"][env_id][:3], reduced_obs)

        if held_object is not None:
            return ToolExecution(poses=[], verifier=None, reason=f"The gripper already holds {held_object}.")

        selected_object = None
        table_pose = extra["table"][env_id]

        for object_name, object_pose in extra.items():
            if not object_name.startswith(f"{product_type}_"):
                continue

            if is_object_inside_target(
                object_pose[env_id],
                table_pose,
                0.3,
                keep_tensor=False,
            ):
                selected_object = object_name
                break

        if selected_object is None:
            return ToolExecution(
                poses=[],
                verifier=None,
                reason=f"No available object of type {product_type} was found.",
            )
        poses = compute_grasp_trajectory(
            self.get_agent(),
            extra[selected_object][env_id].cpu().numpy(),
            add_seuil=0.05,
        )

        def verifier(new_obs: Dict) -> ToolResult:
            new_extra = new_obs["extra"]
            if is_object_in_gripper(
                new_extra["agent_tcp"][env_id],
                new_extra[selected_object][env_id],
                threshold=0.02,
            ):
                return ToolResult(True, reason=f"Successfully took {selected_object}.")

            return ToolResult(False, reason=f"Failed to take an object of type {product_type}.")

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            context={"target_name": selected_object},
            allowed_moving_actors=[selected_object],
        )

    @register_tool(
    description=(
        "Put the held product either on a free table cell "
        "or inside the package currently held by the robot."
    ),
    params_spec={
        "target": {
            "type": str,
            "description": "Destination: table or package.",
        }
    },
    errors=[
        ToolErrorSupport(
            OneShotToolFailureError,
            pre=True,
            post=False,
        )
    ],
    )
    def put(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        if not self._robot_is_at_slot(obs, env_id, "products_slot"):
            return ToolExecution(
                poses=[],
                verifier=None,
                reason="The robot must be at products_slot to put a product.",
            )

        target = params["target"]

        if target not in {"table", "package"}:
            return ToolExecution(poses=[], verifier=None, reason="Unknown target. Use table or package.")

        extra = obs.maniskill_obs["extra"]
        product_types = obs.task_attributes["product_type"]

        product_poses = {
            name: value[env_id][:3]
            for name, value in extra.items()
            if (
                isinstance(value, torch.Tensor)
                and any(
                    name.startswith(f"{product_type}_")
                    for product_type in product_types
                )
            )
        }

        held_object = find_object_in_gripper(
            extra["agent_tcp"][env_id][:3],
            product_poses,
        )

        if held_object is None:
            return ToolExecution(poses=[], verifier=None,reason="The gripper does not hold a product.")

        if target == "package":
            package_name = self._held_package(obs, env_id)

            if package_name is None:
                return ToolExecution(poses=[], verifier=None, reason="The robot does not hold a package.")

            target_pose = extra[package_name]["pose"][env_id]

        else:
            target_cell = self.table_placement_grid.allocate(
                center_pose=extra["table"][env_id],
                object_poses=product_poses,
                batch_context=obs.tool_batch_context,
                owner=held_object,
                reservation_namespace="delivery_box:table",
                excluded_objects=[held_object],
            )
            if target_cell is None:
                return ToolExecution(poses=[], verifier=None, reason="No free cell is available on the table.")
            target_pose = extra["table"][env_id].clone()
            target_pose[:3] = target_cell.world_position
            target_pose[2] += 0.02

        poses = compute_drop_trajectory(
            self.get_agent(),
            drop_pose=target_pose.cpu().numpy(),
            drop_seuil=0.03,
            approach_seuil=0.1,
        )

        def verifier(new_obs: Dict) -> ToolResult:
            new_extra = new_obs["extra"]
            object_pose = new_extra[held_object][env_id]

            if is_object_in_gripper(new_extra["agent_tcp"][env_id], object_pose):
                return ToolResult(False, reason=f"{held_object} is still in the gripper.")
                

            if target == "package":
                package_pose = new_extra[package_name]["pose"][env_id]

                if not is_object_inside_target(object_pose, package_pose, keep_tensor=False):
                    return ToolResult(False, reason=(f"{held_object} is not inside {package_name}."))

                return ToolResult(True, reason=(f"Successfully put {held_object} inside {package_name}."))

            distance = torch.linalg.vector_norm(object_pose[:2] - target_pose[:2])

            if distance > 0.1:
                return ToolResult(False, reason=(f"{held_object} was not placed in the selected table cell."))

            return ToolResult(True, reason=f"Successfully put {held_object} on the table.")

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            allowed_moving_actors=[held_object],
        )

    @register_tool(
        description="Move the robot to a product or package station.",
        params_spec={
            "target": {
                "type": str,
                "description": "Destination: products_slot or packages_slot.",
            }
        },
    )
    def move_to(self, obs: Observation, env_id: int, params: Dict,) -> ToolExecution:
        target = params["target"]
        valid_targets = {"products_slot", "packages_slot"}

        if target not in valid_targets:
            return ToolExecution(poses=[], verifier=None, reason=(f"Unknown target {target}. "))

        extra = obs.maniskill_obs["extra"]

        if target not in extra:
            return ToolExecution(
                poses=[],
                verifier=None,
                reason=f"The target {target} is missing from the observation.",
            )

        robot = self.get_agent()
        robot_state = robot.robot.get_state()[env_id].clone()
        target_pose = extra[target][env_id]

        distance = torch.linalg.vector_norm(
            robot_state[:3] - target_pose[:3]
        )

        if distance < 0.05:
            return ToolExecution(poses=[], verifier=None, reason=f"The robot is already at {target}.")

        robot_state[:3] = target_pose[:3]
        robot_state[3:7] = target_pose[3:7]
        robot_state[7:13] = 0

        state_updates = [
            EnvStateUpdate(
                path=("articulations", robot.robot.name),
                value=robot_state,
            )
        ]
       
        held_package = self._held_package(obs, env_id)

        if held_package is not None:
            old_package_pose = extra[held_package]["pose"][env_id]
            target_package_xyz = HELD_PACKAGE_POSITIONS[target].to(
                device=old_package_pose.device,
                dtype=old_package_pose.dtype,
            )

            package_delta = (target_package_xyz - old_package_pose[:3])

            state_updates.append(
                EnvStateUpdate(
                    path=("articulations", held_package),
                    value=self._package_state_at(
                        extra,
                        env_id,
                        held_package,
                        target_package_xyz,
                    ),
                )
            )

            for object_name in self._objects_in_package(obs, env_id, held_package):
                object_pose = extra[object_name][env_id]

                object_state = torch.zeros(
                    13,
                    device=object_pose.device,
                    dtype=object_pose.dtype,
                )
                object_state[:3] = object_pose[:3] + package_delta
                object_state[3:7] = object_pose[3:7]

                state_updates.append(EnvStateUpdate(path=("actors", object_name), value=object_state))

        def verifier(new_obs: Dict) -> ToolResult:
            return ToolResult(True, reason=f"The robot moved to {target}.", state_updates=state_updates,)

        return ToolExecution(poses=["OK"], verifier=verifier,)

    @register_tool(
        description=(
            "Assign a manufacturing order to a package."
        ),
        params_spec={
            "package": {"type": str,"description": "Package name."},
            "manufacturing_order": {"type": str, "description": "Manufacturing-order reference."},
        },
    )
    def mark_package(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        package_name = params["package"]
        manufacturing_order = params["manufacturing_order"]

        extra = obs.maniskill_obs["extra"]

        if package_name not in PACKAGE_NAMES:
            return ToolExecution(poses=[], verifier=None, reason=f"Unknown package {package_name}.")

        package_state = int(extra[package_name]["state"][env_id].item())

        if package_state == PACKAGE_HELD:
            return ToolExecution(poses=[], verifier=None,
                reason=(f"{package_name} must be dropped before it can be marked."),
            )

        package_objects = self._objects_in_package(obs, env_id, package_name)

        if not package_objects:
            return ToolExecution(poses=[], verifier=None, reason=(f"{package_name} must contain at least one product."))

        product_counts = self._package_product_counts(obs, env_id, package_name,)

        table_pose = extra["table"][env_id]
        product_types = obs.task_attributes["product_type"]
        product_poses = {
            object_name: object_pose[env_id]
            for object_name, object_pose in extra.items()
            if (
                isinstance(object_pose, torch.Tensor)
                and any(
                    object_name.startswith(f"{product_type}_")
                    for product_type in product_types
                )
            )
        }
        target_cells = self.table_placement_grid.allocate_many(
            center_pose=table_pose,
            object_poses=product_poses,
            batch_context=obs.tool_batch_context,
            owners=package_objects,
            reservation_namespace="delivery_box:table",
            excluded_objects=package_objects,
        )
        if target_cells is None:
            return ToolExecution(poses=[],verifier=None,
                reason=("There are not enough free cells on the table to return all package products."))

        state_updates = []

        for object_name, target_cell in zip(package_objects, target_cells):
            object_pose = extra[object_name][env_id]

            object_state = torch.zeros(13, device=object_pose.device, dtype=object_pose.dtype)

            object_state[:3] = target_cell.world_position
            object_state[2] += 0.02

            object_state[3:7] = object_pose[3:7]

            state_updates.append(EnvStateUpdate(path=("actors", object_name), value=object_state))

        def verifier(new_obs: Dict) -> ToolResult:
            return ToolResult(
                True,
                reason=(
                    f"{package_name} was validated with reference {manufacturing_order}. "
                ),
                state_updates=state_updates,
                logs=Log(
                    content={
                        "package": package_name,
                        "manufacturing_order": manufacturing_order,
                        "products": product_counts,
                    }
                ),
            )

        return ToolExecution(poses=["OK"], verifier=verifier)

    @register_tool(
        description="Take an available package from storage.",
        params_spec={
            "package": {
                "type": str,
                "description": "Package name, from package_1 to package_4.",
            }
        },
    )
    def take_box(self, obs: Observation, env_id: int, params: Dict,) -> ToolExecution:
        if not self._robot_is_at_slot(obs, env_id, "packages_slot"):
            return ToolExecution(
                poses=[],
                verifier=None,
                reason="The robot must be at packages_slot to take a package.",
            )

        extra = obs.maniskill_obs["extra"]
        package_name = params["package"]

        if package_name not in PACKAGE_NAMES:
            return ToolExecution(
                [],
                verifier=None,
                reason=f"Unknown package {package_name}.",
            )

        if self._held_package(obs, env_id) is not None:
            return ToolExecution(
                [],
                verifier=None,
                reason="The robot already holds a package.",
            )

        if int(extra[package_name]["state"][env_id].item()) != PACKAGE_FREE:
            return ToolExecution(
                [],
                verifier=None,
                reason=f"{package_name} is not free.",
            )

        target_xyz = HELD_PACKAGE_POSITIONS["packages_slot"].to(
            device=extra[package_name]["pose"].device,
            dtype=extra[package_name]["pose"].dtype,
        )
        package_delta = target_xyz - extra[package_name]["pose"][env_id][:3]

        state_updates = [
            EnvStateUpdate(
                path=("articulations", package_name),
                value=self._package_state_at(extra, env_id, package_name, target_xyz),
            ),
            EnvStateUpdate(
                path=("magma_extra_state", "package_states", package_name),
                value=torch.tensor(
                    PACKAGE_HELD,
                    device=target_xyz.device,
                    dtype=torch.int32,
                ),
            ),
        ]

        for object_name in self._objects_in_package(obs, env_id, package_name):
            object_pose = extra[object_name][env_id]
            object_state = torch.zeros(
                13,
                device=object_pose.device,
                dtype=object_pose.dtype,
            )
            object_state[:3] = object_pose[:3] + package_delta
            object_state[3:7] = object_pose[3:7]

            state_updates.append(
                EnvStateUpdate(
                    path=("actors", object_name),
                    value=object_state,
                )
            )

        def verifier(new_obs: Dict) -> ToolResult:
            return ToolResult(
                True,
                reason=f"The robot took {package_name}.",
                state_updates=state_updates,
            )

        return ToolExecution(["OK"], verifier=verifier)


    @register_tool(
        description="Drop the package currently held by the robot.",
        params_spec={},
    )
    def drop_box(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        if not self._robot_is_at_slot(obs, env_id, "packages_slot"):
            return ToolExecution(
                poses=[],
                verifier=None,
                reason="The robot must be at packages_slot to drop a package.",
            )

        extra = obs.maniskill_obs["extra"]
        package_name = self._held_package(obs, env_id)

        if package_name is None:
            return ToolExecution(
                [],
                verifier=None,
                reason="The robot does not hold a package.",
            )

        old_pose = extra[package_name]["pose"][env_id]

        target_xyz = PACKAGE_HOME_POSITIONS[package_name].to(
            device=old_pose.device,
            dtype=old_pose.dtype,
        )

        delta = target_xyz - old_pose[:3]

        state_updates = [
            EnvStateUpdate(
                path=("articulations", package_name),
                value=self._package_state_at(extra, env_id, package_name, target_xyz),
            ),
            EnvStateUpdate(
                path=("magma_extra_state", "package_states", package_name),
                value=torch.tensor(PACKAGE_FREE, device=target_xyz.device, dtype=torch.int32),
            ),
        ]

        for object_name in self._objects_in_package(obs, env_id, package_name):
            object_pose = extra[object_name][env_id]
            object_state = torch.zeros(
                13,
                device=object_pose.device,
                dtype=object_pose.dtype,
            )
            object_state[:3] = object_pose[:3] + delta
            object_state[3:7] = object_pose[3:7]

            state_updates.append(
                EnvStateUpdate(
                    path=("actors", object_name),
                    value=object_state,
                )
            )

        def verifier(new_obs: Dict) -> ToolResult:
            return ToolResult(
                True,
                reason=f"The robot dropped {package_name}.",
                state_updates=state_updates,
            )

        return ToolExecution(["OK"], verifier=verifier)

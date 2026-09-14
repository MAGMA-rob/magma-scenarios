from typing import Iterable, Optional

import numpy as np
import sapien
import torch

from magma_core.simulation.data_structures import (
    Log,
    Observation,
    ToolExecution,
    ToolResult,
)
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_core.simulation.utils.gripper_utils import (
    find_object_in_gripper,
    is_object_in_gripper,
)
from magma_scenarios.utils import (
    compute_drop_trajectory,
    compute_grasp_trajectory,
)
from magma_scenarios.templates.tools import PlacementGrid

from .attributes import movable_objects


class TableCleaningToolHelpers:
    table_placement_grid = PlacementGrid(
        name="table_cleaning_table",
        rows=3,
        columns=3,
        cell_spacing=0.11,
        selection_order=[1, 4, 7, 5, 6, 3, 2, 8, 0],
    )
    storage_placement_grid = PlacementGrid(
        name="table_cleaning_storage",
        rows=3,
        columns=1,
        cell_spacing=0.11,
    )
    placement_grids = {
        "table": table_placement_grid,
        "food_storage": storage_placement_grid,
        "dish_storage": storage_placement_grid,
        "drying_zone": storage_placement_grid,
        "sink": storage_placement_grid,
    }

    def _object_poses(self, obs: Observation, env_id: int):
        extra = obs.maniskill_obs["extra"]

        return {
            name: extra[name]["pose"][env_id][:3]
            for name in movable_objects
            if name in extra
        }

    def _tcp_pose(self,obs: Observation, env_id: int, tcp_key: str):
        return obs.maniskill_obs["extra"][tcp_key]["pose"][env_id][:3]

    def _held_object(self, obs: Observation, env_id: int, tcp_key: str) -> Optional[str]:
        return find_object_in_gripper(
            self._tcp_pose(obs, env_id, tcp_key),
            self._object_poses(obs, env_id),
        )

    def _object_location(self, obs: Observation, env_id: int, object_name: str, locations: Iterable[str], threshold: float = 0.2) -> Optional[str]:
        extra = obs.maniskill_obs["extra"]

        if object_name not in extra:
            return None

        object_pose = extra[object_name]["pose"][env_id]

        for location in locations:
            if location not in extra:
                continue

            target_pose = extra[location]["pose"][env_id]

            if is_object_inside_target(object_pose, target_pose, thresh=threshold):
                return location

        return None

    def _target_position(
        self,
        obs: Observation,
        env_id: int,
        target: str,
        moving_object: str,
    ):
        extra = obs.maniskill_obs["extra"]

        if target not in extra:
            return None

        target_pose = extra[target]["pose"][env_id]
        grid = self.placement_grids.get(target)
        if grid is None:
            return target_pose[:3].cpu().numpy()

        target_cell = grid.allocate(
            center_pose=target_pose,
            object_poses=self._object_poses(obs, env_id),
            batch_context=obs.tool_batch_context,
            owner=moving_object,
            reservation_namespace=f"table_cleaning:{target}",
            excluded_objects=[moving_object],
        )
        if target_cell is None:
            return None
        return target_cell.world_position.cpu().numpy()

    def _take_object(
            self,
            obs: Observation,
            env_id: int,
            object_name: str,
            tcp_key: str,
            robot_name: str = "",
            final_pose = None
        ) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]
        object_poses = self._object_poses(obs, env_id)
        tcp_pose = self._tcp_pose(obs, env_id, tcp_key)

        if find_object_in_gripper(tcp_pose, object_poses):
            return ToolExecution(poses=[], verifier=None, reason="The gripper is not empty.",)

        if object_name not in object_poses:
            return ToolExecution( poses=[], verifier=None, reason=f"{object_name} was not detected.",)

        object_pose = extra[object_name]["pose"][env_id]
        object_state = extra[object_name]["state"][env_id]

        poses = compute_grasp_trajectory(self.get_agent(robot_name), object_pose.cpu().numpy(), add_seuil=0.2)

        def verifier(new_obs: dict) -> ToolResult:
            new_extra = new_obs["extra"]

            if object_name not in new_extra:
                return ToolResult(False,reason=f"{object_name} no longer exists.")

            if not is_object_in_gripper(
                new_extra[tcp_key]["pose"][env_id],
                new_extra[object_name]["pose"][env_id],
                threshold=0.02,
            ):
                return ToolResult(False, reason=f"Failed to pick up {object_name}.")

            return ToolResult(True,
                reason=f"{object_name} is in the gripper.",
                logs=Log(
                    content={
                        "object": object_name,
                        "state": object_state,
                        "robot": robot_name,
                    }))

        if final_pose:
            poses.append(final_pose)

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            context={"target_name": object_name},
            allowed_moving_actors=[object_name],
        )

    def _put_object(
            self,
            obs: Observation,
            env_id: int,
            target: str,
            tcp_key: str,
            robot_name: str = "",
            container_targets=("trashcan", "washing_machine"),
            final_pose=None
        ) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]
        held_object = self._held_object(obs, env_id, tcp_key)

        if held_object is None:
            return ToolExecution(poses=[], verifier=None, reason="The gripper is empty.")

        if target not in extra:
            return ToolExecution( poses=[], verifier=None, reason=f"Unknown target: {target}." )

        target_position = self._target_position(
            obs,
            env_id,
            target,
            moving_object=held_object,
        )

        if target_position is None:
            return ToolExecution( poses=[], verifier=None, reason=f"There is no free space in {target}.")

        tcp_pose = self._tcp_pose(obs, env_id, tcp_key)
        object_state = extra[held_object]["state"][env_id]

        if target in container_targets:
            drop_seuil = 0.2
            approach_seuil = 0.35
        else:
            drop_seuil = 0.1
            approach_seuil = 0.2

        poses = [
            sapien.Pose(
                p=tcp_pose.cpu().numpy() + np.array([0, 0, 0.1]),
                q=(0, 1, 0, 0),
            )
        ]

        poses.extend(
            compute_drop_trajectory(
                self.get_agent(robot_name),
                drop_pose=target_position,
                drop_seuil=drop_seuil,
                approach_seuil=approach_seuil,
                final_pose=final_pose,
            )
        )
        
        def verifier(new_obs: dict) -> ToolResult:
            new_extra = new_obs["extra"]
            object_pose = new_extra[held_object]["pose"][env_id]

            if is_object_in_gripper(new_extra[tcp_key]["pose"][env_id], object_pose):
                return ToolResult(False, reason=f"{held_object} is still in the gripper.")

            target_pose = new_extra[target]["pose"][env_id]

            if not is_object_inside_target(object_pose, target_pose, thresh=0.2,):
                return ToolResult(False, reason=f"{held_object} is not in {target}.")

            return ToolResult(
                True,
                reason=f"Successfully placed {held_object} in {target}.",
                logs=Log(
                    content={
                        "object": held_object,
                        "state": object_state,
                        "target": target,
                        "robot": robot_name,
                    }
                ),
            )

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            allowed_moving_actors=[held_object],
        )

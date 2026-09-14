# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from dataclasses import dataclass
from typing import Collection, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from magma_core.simulation.data_structures import ToolBatchContext


PoseValue = Union[torch.Tensor, np.ndarray, Sequence[float]]


@dataclass(frozen=True)
class PlacementCell:
    """A selected grid cell expressed locally and in world coordinates."""

    index: int
    local_offset: Tuple[float, float]
    world_position: torch.Tensor


class PlacementGrid:
    """Regular planar grid used to allocate collision-free drop positions."""

    def __init__(
        self,
        name: str,
        rows: int,
        columns: int,
        cell_spacing: Union[float, Tuple[float, float]],
        selection_order: Optional[Sequence[int]] = None,
        occupancy_tolerance: float = 0.02,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("PlacementGrid name must be a non-empty string")
        if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
            raise ValueError("PlacementGrid rows must be a positive integer")
        if not isinstance(columns, int) or isinstance(columns, bool) or columns <= 0:
            raise ValueError("PlacementGrid columns must be a positive integer")

        if isinstance(cell_spacing, (int, float)) and not isinstance(cell_spacing, bool):
            spacing_x = float(cell_spacing)
            spacing_y = float(cell_spacing)
        else:
            if len(cell_spacing) != 2:
                raise ValueError("PlacementGrid cell_spacing must contain two values")
            spacing_x = float(cell_spacing[0])
            spacing_y = float(cell_spacing[1])
        if spacing_x <= 0 or spacing_y <= 0:
            raise ValueError("PlacementGrid cell spacing must be positive")
        if occupancy_tolerance < 0:
            raise ValueError("PlacementGrid occupancy_tolerance cannot be negative")

        self.name = name.strip()
        self.rows = rows
        self.columns = columns
        self.spacing_x = spacing_x
        self.spacing_y = spacing_y
        self.occupancy_tolerance = float(occupancy_tolerance)
        self.cell_offsets = tuple(
            (
                (column - (columns - 1) / 2) * spacing_x,
                (row - (rows - 1) / 2) * spacing_y,
            )
            for row in range(rows)
            for column in range(columns)
        )

        cell_count = rows * columns
        if selection_order is None:
            self.selection_order = tuple(range(cell_count))
        else:
            normalized_order = tuple(selection_order)
            if (
                len(normalized_order) != cell_count
                or set(normalized_order) != set(range(cell_count))
            ):
                raise ValueError(
                    "PlacementGrid selection_order must contain every cell index exactly once"
                )
            self.selection_order = normalized_order

    @staticmethod
    def _rotate(vector: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
        quaternion_norm = torch.linalg.vector_norm(quaternion)
        if float(quaternion_norm) == 0:
            raise ValueError("PlacementGrid support quaternion cannot be zero")
        quaternion = quaternion / quaternion_norm
        cross = 2 * torch.cross(quaternion[1:], vector, dim=0)
        return vector + quaternion[0] * cross + torch.cross(
            quaternion[1:],
            cross,
            dim=0,
        )

    @staticmethod
    def _pose_tensor(value: PoseValue, reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if reference is None:
                return value if value.is_floating_point() else value.to(dtype=torch.float32)
            return value.to(device=reference.device, dtype=reference.dtype)
        if reference is None:
            return torch.as_tensor(value, dtype=torch.float32)
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)

    def occupied_cells(
        self,
        center_pose: PoseValue,
        object_poses: Mapping[str, PoseValue],
        excluded_objects: Collection[str] = (),
    ) -> Dict[int, list[str]]:
        """Return physical occupants grouped by their nearest grid cell."""
        center = self._pose_tensor(center_pose)
        if center.numel() < 3:
            raise ValueError("PlacementGrid center_pose must contain at least xyz")
        quaternion = (
            center[3:7]
            if center.numel() >= 7
            else torch.tensor(
                [1, 0, 0, 0],
                device=center.device,
                dtype=center.dtype,
            )
        )
        inverse_quaternion = quaternion.clone()
        inverse_quaternion[1:] *= -1

        half_width = max(abs(offset[0]) for offset in self.cell_offsets) + self.spacing_x / 2
        half_height = max(abs(offset[1]) for offset in self.cell_offsets) + self.spacing_y / 2
        excluded = set(excluded_objects)
        occupied: Dict[int, list[str]] = {}

        for actor_name, actor_pose in object_poses.items():
            if actor_name in excluded:
                continue
            actor = self._pose_tensor(actor_pose, center)
            if actor.numel() < 3:
                raise ValueError(
                    f"PlacementGrid actor pose for {actor_name!r} must contain at least xyz"
                )
            local_position = self._rotate(actor[:3] - center[:3], inverse_quaternion)
            if (
                abs(float(local_position[0])) > half_width + self.occupancy_tolerance
                or abs(float(local_position[1])) > half_height + self.occupancy_tolerance
            ):
                continue

            cell_index = min(
                range(len(self.cell_offsets)),
                key=lambda index: (
                    (float(local_position[0]) - self.cell_offsets[index][0]) ** 2
                    + (float(local_position[1]) - self.cell_offsets[index][1]) ** 2
                ),
            )
            occupied.setdefault(cell_index, []).append(actor_name)

        return occupied

    def allocate(
        self,
        center_pose: PoseValue,
        object_poses: Mapping[str, PoseValue],
        batch_context: ToolBatchContext,
        owner: str,
        reservation_namespace: Optional[str] = None,
        excluded_objects: Collection[str] = (),
    ) -> Optional[PlacementCell]:
        """Select and reserve the first physically and logically free cell."""
        center = self._pose_tensor(center_pose)
        quaternion = (
            center[3:7]
            if center.numel() >= 7
            else torch.tensor(
                [1, 0, 0, 0],
                device=center.device,
                dtype=center.dtype,
            )
        )
        occupied = self.occupied_cells(
            center,
            object_poses,
            excluded_objects,
        )
        namespace = self.name if reservation_namespace is None else reservation_namespace
        reserved = batch_context.get_reservations(namespace)

        for cell_index in self.selection_order:
            if cell_index in occupied or cell_index in reserved:
                continue
            if not batch_context.try_reserve(namespace, cell_index, owner):
                continue

            offset_x, offset_y = self.cell_offsets[cell_index]
            local_offset = torch.tensor(
                [offset_x, offset_y, 0],
                device=center.device,
                dtype=center.dtype,
            )
            world_position = center[:3] + self._rotate(local_offset, quaternion)
            return PlacementCell(
                index=cell_index,
                local_offset=(offset_x, offset_y),
                world_position=world_position,
            )

        return None

    def allocate_many(
        self,
        center_pose: PoseValue,
        object_poses: Mapping[str, PoseValue],
        batch_context: ToolBatchContext,
        owners: Sequence[str],
        reservation_namespace: Optional[str] = None,
        excluded_objects: Collection[str] = (),
    ) -> Optional[Tuple[PlacementCell, ...]]:
        """Reserve one free cell per owner without partial allocation."""
        normalized_owners = tuple(owners)
        if not normalized_owners:
            return ()

        center = self._pose_tensor(center_pose)
        quaternion = (
            center[3:7]
            if center.numel() >= 7
            else torch.tensor(
                [1, 0, 0, 0],
                device=center.device,
                dtype=center.dtype,
            )
        )
        occupied = self.occupied_cells(
            center,
            object_poses,
            excluded_objects,
        )
        namespace = self.name if reservation_namespace is None else reservation_namespace
        reserved = batch_context.get_reservations(namespace)
        available_indices = [
            cell_index
            for cell_index in self.selection_order
            if cell_index not in occupied and cell_index not in reserved
        ]
        if len(available_indices) < len(normalized_owners):
            return None

        selected_indices = available_indices[:len(normalized_owners)]
        cells = []
        for owner, cell_index in zip(normalized_owners, selected_indices):
            if not batch_context.try_reserve(namespace, cell_index, owner):
                raise RuntimeError(
                    "PlacementGrid reservation changed during an atomic allocation"
                )

            offset_x, offset_y = self.cell_offsets[cell_index]
            local_offset = torch.tensor(
                [offset_x, offset_y, 0],
                device=center.device,
                dtype=center.dtype,
            )
            cells.append(PlacementCell(
                index=cell_index,
                local_offset=(offset_x, offset_y),
                world_position=center[:3] + self._rotate(local_offset, quaternion),
            ))

        return tuple(cells)

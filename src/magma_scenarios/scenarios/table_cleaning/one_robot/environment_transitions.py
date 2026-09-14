from typing import Dict, Iterable, List, Sequence

import torch

from magma_core.simulation.stage import BaseStageEnvironmentTransition

from ..common.attributes import CLEAN_STATE, DIRTY_STATE, food, cleaning_objects
from ..common.layout import (
    DISH_STORAGE_CENTER,
    FOOD_STORAGE_CENTER,
    STORAGE_DISTANCE_THRESHOLD,
    STORAGE_OFFSETS,
    TABLE_CENTER,
    TABLE_DISTANCE_THRESHOLD,
)


def _is_near(position: torch.Tensor, center: Sequence[float], threshold: float) -> bool:
    reference = torch.tensor(
        center,
        device=position.device,
        dtype=position.dtype,
    )
    return bool(torch.linalg.vector_norm(position[:2] - reference) <= threshold)


class DirtyTableObjectsTransition(BaseStageEnvironmentTransition):
    def __init__(self, dirty_objects: Iterable[str]) -> None:
        self.dirty_objects = list(dirty_objects)

    def _to_spec_arguments(self) -> Dict:
        return {"dirty_objects": self.dirty_objects.copy()}

    def apply(self, env_state: Dict, env_ids: torch.Tensor) -> Dict:
        actors = env_state.get("actors", {})
        object_states = env_state.get("magma_extra_state", {})

        for env_id in env_ids.tolist():
            for object_name in self.dirty_objects:
                if object_name not in actors or object_name not in object_states:
                    continue
                if not _is_near(
                    actors[object_name][env_id, :3],
                    TABLE_CENTER,
                    TABLE_DISTANCE_THRESHOLD,
                ):
                    continue
                object_states[object_name][env_id] = DIRTY_STATE

        return env_state


class RestockStorageTransition(BaseStageEnvironmentTransition):
    def __init__(self, active_objects: Iterable[str]) -> None:
        self.active_objects = [
            object_name
            for object_name in active_objects
            if object_name not in cleaning_objects
        ]

    def _to_spec_arguments(self) -> Dict:
        return {"active_objects": self.active_objects.copy()}

    def _restock_group(
        self,
        actors: Dict,
        object_states: Dict,
        env_id: int,
        object_names: List[str],
        center: Sequence[float],
    ) -> None:
        cells = [
            (center[0] + offset[0], center[1] + offset[1])
            for offset in STORAGE_OFFSETS
        ]
        present = []
        to_move = []

        for object_name in object_names:
            if object_name not in actors or object_name not in object_states:
                continue
            position = actors[object_name][env_id, :3]
            if _is_near(position, center, STORAGE_DISTANCE_THRESHOLD):
                present.append(object_name)
            else:
                to_move.append(object_name)
            object_states[object_name][env_id] = CLEAN_STATE

        available_cells = cells.copy()
        for object_name in present:
            position = actors[object_name][env_id, :2]
            closest_cell = min(
                available_cells,
                key=lambda cell: float(
                    torch.linalg.vector_norm(
                        position
                        - torch.tensor(
                            cell,
                            device=position.device,
                            dtype=position.dtype,
                        )
                    )
                ),
            )
            available_cells.remove(closest_cell)

        if len(to_move) > len(available_cells):
            raise RuntimeError("Not enough storage cells to restock table-cleaning objects.")

        for object_name, cell in zip(to_move, available_cells):
            actor_state = actors[object_name][env_id]
            actor_state[:3] = torch.tensor(
                [cell[0], cell[1], 0.01],
                device=actor_state.device,
                dtype=actor_state.dtype,
            )
            actor_state[7:] = 0

    def apply(self, env_state: Dict, env_ids: torch.Tensor) -> Dict:
        actors = env_state.get("actors", {})
        object_states = env_state.get("magma_extra_state", {})

        food_objects = [
            object_name
            for object_name in self.active_objects
            if object_name in food
        ]
        dish_objects = [
            object_name
            for object_name in self.active_objects
            if object_name not in food
        ]

        for env_id in env_ids.tolist():
            self._restock_group(
                actors,
                object_states,
                env_id,
                food_objects,
                FOOD_STORAGE_CENTER,
            )
            self._restock_group(
                actors,
                object_states,
                env_id,
                dish_objects,
                DISH_STORAGE_CENTER,
            )

        return env_state

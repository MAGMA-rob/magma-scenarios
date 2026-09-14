from typing import Dict, Iterable

import torch

from magma_core.simulation.stage import BaseStageEnvironmentTransition

from ..common.attributes import CLEAN_STATE, food
from ..common.layout import (
    FOOD_STORAGE_CENTER,
    STORAGE_DISTANCE_THRESHOLD,
    STORAGE_OFFSETS,
)


def _is_near(position, center, threshold: float) -> bool:
    center_tensor = torch.as_tensor(
        center,
        device=position.device,
        dtype=position.dtype,
    )
    return bool(
        torch.linalg.vector_norm(position[:2] - center_tensor[:2])
        <= threshold
    )


class ResetFoodTransition(BaseStageEnvironmentTransition):
    def __init__(self, active_objects: Iterable[str]) -> None:
        self.food_objects = [
            object_name
            for object_name in active_objects
            if object_name in food
        ]

    def _to_spec_arguments(self) -> Dict:
        return {"active_objects": self.food_objects.copy()}

    def apply(self, env_state: Dict, env_ids: torch.Tensor) -> Dict:
        actors = env_state.get("actors", {})
        object_states = env_state.get("magma_extra_state", {})
        if "trashcan" not in actors:
            return env_state

        storage_cells = [
            (
                FOOD_STORAGE_CENTER[0] + offset[0],
                FOOD_STORAGE_CENTER[1] + offset[1],
            )
            for offset in STORAGE_OFFSETS
        ]

        for env_id in env_ids.tolist():
            available_cells = storage_cells.copy()
            discarded_food = []
            trashcan_position = actors["trashcan"][env_id, :3]

            for object_name in self.food_objects:
                if object_name not in actors or object_name not in object_states:
                    continue

                object_states[object_name][env_id] = CLEAN_STATE
                position = actors[object_name][env_id, :3]
                if _is_near(
                    position,
                    FOOD_STORAGE_CENTER,
                    STORAGE_DISTANCE_THRESHOLD,
                ):
                    closest_cell = min(
                        available_cells,
                        key=lambda cell: float(
                            torch.linalg.vector_norm(
                                position[:2]
                                - torch.tensor(
                                    cell,
                                    device=position.device,
                                    dtype=position.dtype,
                                )
                            )
                        ),
                    )
                    available_cells.remove(closest_cell)
                elif _is_near(
                    position,
                    trashcan_position,
                    0.2,
                ):
                    discarded_food.append(object_name)

            if len(discarded_food) > len(available_cells):
                raise RuntimeError(
                    "Not enough food-storage cells to reset discarded food."
                )

            for object_name, cell in zip(discarded_food, available_cells):
                actor_state = actors[object_name][env_id]
                actor_state[:3] = torch.tensor(
                    [cell[0], cell[1], 0.01],
                    device=actor_state.device,
                    dtype=actor_state.dtype,
                )
                actor_state[7:] = 0

        return env_state

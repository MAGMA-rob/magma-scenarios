import random
from typing import Dict, Mapping, Optional, Tuple

import torch

from magma_core.simulation.stage import BaseStageEnvironmentTransition

from .attributes import DRINKS, FOOD_OBJECTS, FRUITS


TABLE_CENTER = (-0.1, -0.2)
GRID_STEP = 0.11
TABLE_CELLS = [
    (TABLE_CENTER[0] + x_offset, TABLE_CENTER[1] + y_offset)
    for x_offset in (-GRID_STEP, 0.0, GRID_STEP)
    for y_offset in (-GRID_STEP, 0.0, GRID_STEP)
]


class ResetPackagingTransition(BaseStageEnvironmentTransition):
    """Return every food object to one serialized table-grid cell."""

    def __init__(
        self,
        object_cells: Optional[Mapping[str, Tuple[float, float]]] = None,
    ) -> None:
        if object_cells is None:
            shuffled_cells = random.sample(TABLE_CELLS, k=len(FOOD_OBJECTS))
            object_cells = dict(zip(FOOD_OBJECTS, shuffled_cells))

        self.object_cells = {
            object_name: tuple(cell)
            for object_name, cell in object_cells.items()
        }
        if set(self.object_cells) != set(FOOD_OBJECTS):
            raise ValueError(
                "object_cells must define exactly one cell for every food object"
            )
        if len(set(self.object_cells.values())) != len(FOOD_OBJECTS):
            raise ValueError("Packaging reset cells must be unique")
        if any(cell not in TABLE_CELLS for cell in self.object_cells.values()):
            raise ValueError("object_cells contains a position outside the table grid")

    def _to_spec_arguments(self) -> Dict:
        return {"object_cells": self.object_cells.copy()}

    def apply(self, env_state: Dict, env_ids: torch.Tensor) -> Dict:
        actors = env_state.get("actors", {})
        missing_objects = [food for food in FOOD_OBJECTS if food not in actors]
        if missing_objects:
            raise RuntimeError(
                f"Packaging environment is missing actors: {missing_objects}"
            )

        for env_id in env_ids.tolist():
            for food, cell in self.object_cells.items():
                actor_state = actors[food][env_id]
                if food in FRUITS:
                    height = 0.018
                elif food in DRINKS:
                    height = 0.02
                else:
                    height = 0.025

                actor_state[:3] = torch.tensor(
                    [cell[0], cell[1], height],
                    device=actor_state.device,
                    dtype=actor_state.dtype,
                )
                actor_state[3:7] = torch.tensor(
                    [1, 0, 0, 0],
                    device=actor_state.device,
                    dtype=actor_state.dtype,
                )
                actor_state[7:] = 0

        return env_state

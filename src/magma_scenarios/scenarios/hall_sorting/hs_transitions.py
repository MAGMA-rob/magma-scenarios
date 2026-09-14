import random
from typing import Dict, List
import torch
from magma_core.simulation.stage import BaseStageEnvironmentTransition
from .attributes import HALLS, OBJECT_TYPES


HALL_CAPACITY = 4
OBJECT_HEIGHT = 0.02

HALL_CENTERS = {
    "Hall1": (-1.85, 0.0),
    "Hall2": (0.15, 2.0),
    "Hall3": (2.15, 0.0),
    "Hall4": (0.15, -2.0),
}

HALL_OFFSETS = [
    (-0.10, -0.10),
    (-0.10, 0.10),
    (0.10, -0.10),
    (0.10, 0.10),
]


class ResetHallDeliveryTransition(BaseStageEnvironmentTransition):
    def __init__(self, delivery_layout: Dict[str, List[str]]) -> None:
        self.delivery_layout = {
            hall: list(delivery_layout.get(hall, []))
            for hall in HALLS
        }

        assigned_objects = [
            object_name
            for objects in self.delivery_layout.values()
            for object_name in objects
        ]

        if len(assigned_objects) != len(set(assigned_objects)):
            raise ValueError("An object cannot be assigned to multiple halls.")

        for hall, objects in self.delivery_layout.items():
            if len(objects) > HALL_CAPACITY:
                raise ValueError(
                    f"{hall} cannot contain more than {HALL_CAPACITY} objects."
                )

    def _to_spec_arguments(self) -> Dict:
        return {
            "delivery_layout": {
                hall: objects.copy()
                for hall, objects in self.delivery_layout.items()
            }
        }

    @staticmethod
    def _is_product(object_name: str) -> bool:
        return any(
            object_name.startswith(f"{object_type}_")
            for object_type in OBJECT_TYPES
        )

    def apply(self, env_state: Dict, env_ids: torch.Tensor,) -> Dict:
        actors = env_state.get("actors", {})

        assigned_objects = {
            object_name
            for objects in self.delivery_layout.values()
            for object_name in objects
        }

        existing_objects = {
            object_name
            for object_name in actors
            if self._is_product(object_name)
        }

        if assigned_objects != existing_objects:
            missing_objects = sorted(existing_objects - assigned_objects)
            unknown_objects = sorted(assigned_objects - existing_objects)

            raise RuntimeError(
                "The delivery layout must contain every active "
                f"object exactly once. Missing: {missing_objects}; "
                f"unknown: {unknown_objects}."
            )

        for env_id in env_ids.tolist():
            for hall, object_names in self.delivery_layout.items():
                center_x, center_y = HALL_CENTERS[hall]

                offsets = random.sample(HALL_OFFSETS, k=len(object_names),)

                for object_name, offset in zip(object_names, offsets,):
                    actor_state = actors[object_name][env_id]

                    actor_state[..., :3] = torch.tensor(
                        [
                            center_x + offset[0],
                            center_y + offset[1],
                            OBJECT_HEIGHT,
                        ],
                        device=actor_state.device,
                        dtype=actor_state.dtype,
                    )

                    actor_state[..., 3:7] = torch.tensor(
                        [1, 0, 0, 0],
                        device=actor_state.device,
                        dtype=actor_state.dtype,
                    )

                    actor_state[..., 7:] = 0
                    
        return env_state

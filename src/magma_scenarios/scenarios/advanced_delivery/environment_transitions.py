import random
from typing import Dict, Iterable, List, Tuple
import torch
from magma_core.simulation.stage import BaseStageEnvironmentTransition
from .attributes import PRODUCT_TYPES
from .utils import product_type_from_name


CLEAN_STATE = 1
BROKEN_STATE = 0
OBJECT_HEIGHT = 0.02

RECEPTION_CENTER = (-1.85, -1.65)

RECEPTION_OFFSETS = [
    (-0.10, -0.10),(0.00, -0.10),(0.10, -0.10),
    (-0.10, 0.00),(0.00, 0.00),(0.10, 0.00),
    (-0.10, 0.10),(0.00, 0.10),(0.10, 0.10),
]

STORAGE_CENTERS = {
    "electronics": (1.15, 1.70),
    "drinks": (1.15, 0.85),
    "snacks": (1.15, 0.00),
    "hygiene": (1.15, -0.85),
    "textile": (1.15, -1.70),
}

STORAGE_OFFSETS = [
    (-0.06, -0.06),
    (0.06, -0.06),
    (-0.06, 0.06),
    (0.06, 0.06),
]


def _storage_cells(product_type: str) -> List[Tuple[float, float]]:
    center = STORAGE_CENTERS[product_type]

    return [
        (
            center[0] + offset[0],
            center[1] + offset[1],
        )
        for offset in STORAGE_OFFSETS
    ]


def _move_object(
    actors: Dict,
    env_id: int,
    object_name: str,
    target_position: Tuple[float, float, float],
) -> None:
    actor_state = actors[object_name][env_id]

    actor_state[..., :3] = torch.tensor(
        target_position,
        device=actor_state.device,
        dtype=actor_state.dtype,
    )

    actor_state[..., 7:] = 0


def _product_objects(actors: Dict) -> List[str]:
    return sorted(
        object_name
        for object_name in actors
        if product_type_from_name(object_name) is not None
    )


class ResetAdvancedDeliveryTransition(BaseStageEnvironmentTransition):
    """
    Reset all products:

    - selected reception objects go to the reception table;
    - selected broken objects are marked as broken;
    - every other product returns clean to its storage.
    """

    def __init__(
        self,
        reception_objects: Iterable[str],
        broken_objects: Iterable[str],
        reception_offsets: Iterable[Tuple[float, float]] | None = None,
    ) -> None:
        self.reception_objects = list(
            reception_objects
        )
        self.broken_objects = list(
            broken_objects
        )

        if (
            len(set(self.reception_objects))
            != len(self.reception_objects)
        ):
            raise ValueError(
                "reception_objects contains duplicate instances."
            )

        if (
            len(set(self.broken_objects))
            != len(self.broken_objects)
        ):
            raise ValueError(
                "broken_objects contains duplicate instances."
            )

        if len(self.reception_objects) > len(
            RECEPTION_OFFSETS
        ):
            raise ValueError(
                "The reception area cannot contain more than "
                f"{len(RECEPTION_OFFSETS)} products."
            )

        unknown_broken_objects = (
            set(self.broken_objects)
            - set(self.reception_objects)
        )

        if unknown_broken_objects:
            raise ValueError(
                "Broken objects must also be reception "
                f"objects: {sorted(unknown_broken_objects)}."
            )

        for object_name in self.reception_objects:
            if product_type_from_name(object_name) is None:
                raise ValueError(
                    f"Invalid product instance: "
                    f"{object_name!r}."
                )

        if reception_offsets is None:
            self.reception_offsets = random.sample(
                RECEPTION_OFFSETS,
                k=len(self.reception_objects),
            )
        else:
            self.reception_offsets = list(reception_offsets)
            if len(self.reception_offsets) != len(self.reception_objects):
                raise ValueError(
                    "reception_offsets must define one offset per reception object."
                )
            if any(offset not in RECEPTION_OFFSETS for offset in self.reception_offsets):
                raise ValueError("reception_offsets contains an unknown cell.")
            if len(set(self.reception_offsets)) != len(self.reception_offsets):
                raise ValueError("reception_offsets contains duplicate cells.")

    def _to_spec_arguments(self) -> Dict:
        return {
            "reception_objects": self.reception_objects.copy(),
            "broken_objects": self.broken_objects.copy(),
            "reception_offsets": self.reception_offsets.copy(),
        }

    def apply(self,env_state: Dict,env_ids: torch.Tensor) -> Dict:
        actors = env_state.get("actors", {})

        object_states = env_state.get(
            "magma_extra_state",
            {},
        ).get(
            "object_states",
            {},
        )

        product_objects = _product_objects(actors)

        unknown_reception_objects = (
            set(self.reception_objects)
            - set(product_objects)
        )

        if unknown_reception_objects:
            raise RuntimeError(
                "Unknown reception product instances: "
                f"{sorted(unknown_reception_objects)}."
            )

        missing_states = [
            object_name
            for object_name in product_objects
            if object_name not in object_states
        ]

        if missing_states:
            raise RuntimeError(
                "Missing product states for: "
                f"{missing_states}."
            )

        reception_set = set(
            self.reception_objects
        )

        broken_set = set(
            self.broken_objects
        )

        storage_objects_by_type = {
            product_type: []
            for product_type in PRODUCT_TYPES
        }

        for object_name in product_objects:
            if object_name in reception_set:
                continue

            product_type = product_type_from_name(
                object_name
            )

            if product_type is not None:
                storage_objects_by_type[
                    product_type
                ].append(object_name)

        for product_type, object_names in (
            storage_objects_by_type.items()
        ):
            if len(object_names) > len(
                STORAGE_OFFSETS
            ):
                raise RuntimeError(
                    f"Not enough storage cells for "
                    f"{product_type}: "
                    f"{len(object_names)} products for "
                    f"{len(STORAGE_OFFSETS)} cells."
                )

        for env_id in env_ids.tolist():
            # Every product starts the new cycle clean.
            for object_name in product_objects:
                object_states[object_name][
                    env_id
                ] = CLEAN_STATE

            # Return every non-reception product to its storage.
            for product_type, object_names in (
                storage_objects_by_type.items()
            ):
                cells = _storage_cells(product_type)

                for object_name, cell in zip(
                    object_names,
                    cells,
                ):
                    _move_object(
                        actors=actors,
                        env_id=env_id,
                        object_name=object_name,
                        target_position=(
                            cell[0],
                            cell[1],
                            OBJECT_HEIGHT,
                        ),
                    )

            # Place the selected products in reception.
            for object_name, offset in zip(
                self.reception_objects,
                self.reception_offsets,
            ):
                _move_object(
                    actors=actors,
                    env_id=env_id,
                    object_name=object_name,
                    target_position=(
                        RECEPTION_CENTER[0]
                        + offset[0],
                        RECEPTION_CENTER[1]
                        + offset[1],
                        OBJECT_HEIGHT,
                    ),
                )

            # Mark only the selected reception products as broken.
            for object_name in broken_set:
                object_states[object_name][
                    env_id
                ] = BROKEN_STATE

        return env_state

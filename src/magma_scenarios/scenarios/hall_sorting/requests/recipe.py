
import random
from collections import Counter
from copy import deepcopy
from typing import Dict, List
from magma_core.simulation.state import TaskState
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.requests import BaseRequest
from magma_core.utils.text_utils import join_with_and
from .deliveries import (
    DeliveryCycleParameters,
    build_delivery_stages,
    freeze_layout,
    freeze_typed_assignment,
    thaw_layout,
    thaw_typed_assignment,
)
from ..hs_transitions import ResetHallDeliveryTransition
from .common import (
    PRIORITY_HALL_NEEDS_APPLICATION_KEY,
    TYPE_HALL_NEEDS_APPLICATION_KEY,
    get_available_delivery_types,
    get_objects_by_type,
    DELIVERY_LAYOUT_KEY,
    sample_next_delivery_layout,
)

RECIPE_WEIGHT = 1.5
HALL_CAPACITY = 4


def _format_product_quantity(object_type: str, quantity: int) -> str:
    if quantity == 1:
        return f"1 {object_type}"

    return f"{quantity} {object_type}s"


def _build_recipe_description(recipe: Dict[str, int]) -> str:
    parts = [
        _format_product_quantity(
            object_type=object_type,
            quantity=quantity,
        )
        for object_type, quantity in recipe.items()
        if quantity > 0
    ]

    return join_with_and(parts)


class DeliverRecipeRequest(BaseRequest[DeliveryCycleParameters]):
    """
    Deliver explicitly requested object quantities to one storage hall.

    The recipe is temporary and does not modify the permanent
    type_hall relation.
    """

    def __init__(
        self,
        max_recipe_types: int = 3,
        max_objects: int = HALL_CAPACITY,
        sampling_weight: float = RECIPE_WEIGHT,
    ) -> None:
        super().__init__()

        if max_recipe_types < 1:
            raise ValueError("max_recipe_types must be at least 1.")

        if max_objects < 1:
            raise ValueError("max_objects must be at least 1.")

        if max_objects > HALL_CAPACITY:
            raise ValueError(f"max_objects cannot exceed the hall capacity of {HALL_CAPACITY}.")

        if max_recipe_types > max_objects:
            raise ValueError("max_recipe_types cannot exceed max_objects.")

        if sampling_weight < 0:
            raise ValueError("sampling_weight must be non-negative.")

        self.max_recipe_types = max_recipe_types
        self.max_objects = max_objects
        self.weight = sampling_weight

    def sampling_weight(self, state: TaskState) -> float:
        storage_halls = state.attributes.get("storage_halls", [])

        if not storage_halls:
            return 0

        if not get_objects_by_type(state):
            return 0

        # Newly created type-to-hall rules should be tested before
        # introducing an unrelated explicit recipe.
        if state.properties.get(
            TYPE_HALL_NEEDS_APPLICATION_KEY,
            False,
        ):
            return 0

        # A new priority must first be exercised by a complete
        # delivery cycle.
        if state.properties.get(
            PRIORITY_HALL_NEEDS_APPLICATION_KEY,
            False,
        ):
            return 0

        return self.weight

    def _sample_recipe(self, state: TaskState) -> Dict[str, int]:
        objects_by_type = get_objects_by_type(state)

        available_types = get_available_delivery_types(state)

        if not available_types:
            raise RuntimeError("Cannot build a recipe without available objects.")

        type_count = random.randint(
            1,
            min(
                self.max_recipe_types,
                self.max_objects,
                len(available_types),
            ),
        )

        selected_types = random.sample(
            available_types,
            type_count,
        )

        # Start with one instance of every selected type, ensuring that
        # each type mentioned in the recipe is actually represented.
        selected_objects = [
            random.choice(objects_by_type[object_type])
            for object_type in selected_types
        ]

        remaining_objects: List[str] = []

        for object_type in selected_types:
            selected_for_type = {
                object_name
                for object_name in selected_objects
                if object_name.startswith(
                    f"{object_type}_"
                )
            }

            remaining_objects.extend(
                object_name
                for object_name in objects_by_type[object_type]
                if object_name not in selected_for_type
            )

        random.shuffle(remaining_objects)

        maximum_extra_count = min(
            self.max_objects - len(selected_objects),
            len(remaining_objects),
        )

        extra_count = random.randint(
            0,
            maximum_extra_count,
        )

        selected_objects.extend(
            remaining_objects[:extra_count]
        )

        selected_object_types = [
            object_name.rsplit("_", 1)[0]
            for object_name in selected_objects
        ]

        return dict(
            Counter(selected_object_types)
        )

    def _build_assignment(
        self,
        state: TaskState,
        target_hall: str,
        recipe: Dict[str, int],
    ) -> Dict[str, Dict[str, int]]:
        storage_halls = state.attributes.get("storage_halls", [])

        known_types = state.attributes.get("object_classes", [])

        assignment = {
            hall: {
                object_type: 0
                for object_type in known_types
            }
            for hall in storage_halls
        }

        if target_hall not in assignment:
            raise ValueError(f"{target_hall!r} is not a storage hall.")

        for object_type, quantity in recipe.items():
            if object_type not in known_types:
                raise ValueError(
                    f"Unknown object type: {object_type!r}."
                )

            assignment[target_hall][object_type] = quantity

        return assignment

    def sample_parameters(self, state: TaskState) -> DeliveryCycleParameters:
        current_layout = state.properties.get(DELIVERY_LAYOUT_KEY, {})

        reception_halls = state.attributes.get(
            "reception_halls",
            [],
        )

        next_delivery_layout = sample_next_delivery_layout(
            current_layout=current_layout,
            reception_halls=reception_halls,
        )

        delivery_state = deepcopy(state)

        delivery_state.properties[DELIVERY_LAYOUT_KEY
        ] = deepcopy(next_delivery_layout)

        storage_halls = delivery_state.attributes.get("storage_halls",[])

        if not storage_halls:
            raise RuntimeError("No storage hall is available for a recipe.")

        target_hall = random.choice(storage_halls)

        recipe = self._sample_recipe(delivery_state)

        assignment = self._build_assignment(
            state=delivery_state,
            target_hall=target_hall,
            recipe=recipe,
        )

        recipe_description = _build_recipe_description(
            recipe
        )

        instruction = (
            "As a temporary exception to the default "
            f"storage rules, deliver {recipe_description} to {target_hall}."
        )

        objects_by_type = get_objects_by_type(delivery_state)
        object_targets = {
            object_name: target_hall
            for object_type, quantity in recipe.items()
            for object_name in sorted(
                objects_by_type.get(object_type, [])
            )[:quantity]
        }
        return DeliveryCycleParameters(
            freeze_layout(next_delivery_layout),
            (),
            freeze_typed_assignment(assignment),
            instruction,
            object_targets=tuple(object_targets.items()),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: DeliveryCycleParameters,
    ) -> List[BaseTaskStage]:
        stages = build_delivery_stages(
            thaw_typed_assignment(parameters.assignment),
            parameters.instruction,
        )
        stages[0].entry_transition = ResetHallDeliveryTransition(
            delivery_layout=thaw_layout(parameters.next_delivery_layout),
        )
        stages[-1].global_parameters.reset_at_end = False
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: DeliveryCycleParameters,
    ) -> TaskState:
        state.properties[DELIVERY_LAYOUT_KEY] = thaw_layout(
            parameters.next_delivery_layout
        )
        return state

import random
from dataclasses import dataclass
from typing import List, Tuple

from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest

from ..brs_attributes import OBJECT_TYPES, ZONES
from ..brs_stages import SortDamagedObjectsStage
from .common import (
    OBJECT_AREAS_KEY,
    TYPE_PRIORITY_KEY,
    ExactFruitMove,
    FrozenAssignment,
    apply_exact_moves,
    empty_assignment,
    extract_moves,
    format_zone,
    freeze_assignment,
    get_damaged_assignment,
    get_damaged_objects,
    set_damaged_assignment,
    select_exact_moves,
    thaw_assignment,
)


LAST_DAMAGED_TARGET_PROPERTY_KEY = "_last_damaged_target"


@dataclass(frozen=True)
class DamagedSortingParameters:
    current: FrozenAssignment
    target: FrozenAssignment
    damaged_objects: Tuple[str, ...]
    target_zone: str
    exact_moves: Tuple[ExactFruitMove, ...]
    resulting_object_areas: Tuple[Tuple[str, str], ...]
    instruction: str


class SortDamagedObjectsRequest(BaseRequest[DamagedSortingParameters]):
    def __init__(self, weight: float = 2.0) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError("weight must be non-negative.")
        self.weight = weight

    def sampling_weight(self, state: TaskState) -> float:
        if state.properties.get(TYPE_PRIORITY_KEY) is not None:
            return 0
        return self.weight if get_damaged_objects(state) else 0

    def sample_parameters(self, state: TaskState) -> DamagedSortingParameters:
        damaged_objects = get_damaged_objects(state)
        if not damaged_objects:
            raise RuntimeError("No damaged object is available.")
        previous_target = state.properties.get(LAST_DAMAGED_TARGET_PROPERTY_KEY)
        object_areas = state.properties[OBJECT_AREAS_KEY]
        candidate_zones = [
            zone
            for zone in ZONES
            if zone != previous_target
            and any(object_areas[name] != zone for name in damaged_objects)
        ]
        if not candidate_zones:
            raise RuntimeError(
                "No different target is available for the damaged objects."
            )
        target_zone = random.choice(candidate_zones)
        instruction = (
            f"Place all damaged objects in the {format_zone(target_zone)}."
        )
        current = get_damaged_assignment(state)
        target = empty_assignment()
        for object_type in OBJECT_TYPES:
            target[target_zone][object_type] = sum(
                current[zone][object_type] for zone in ZONES
            )
        exact_moves = select_exact_moves(
            object_areas,
            damaged_objects,
            extract_moves(current, target),
            damaged=True,
        )
        return DamagedSortingParameters(
            current=freeze_assignment(current),
            target=freeze_assignment(target),
            damaged_objects=tuple(damaged_objects),
            target_zone=target_zone,
            exact_moves=tuple(exact_moves),
            resulting_object_areas=tuple(
                apply_exact_moves(object_areas, exact_moves).items()
            ),
            instruction=instruction,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: DamagedSortingParameters,
    ) -> List[BaseTaskStage]:
        object_areas = state.properties[OBJECT_AREAS_KEY]
        correctly_placed = sum(
            object_areas[name] == parameters.target_zone
            for name in parameters.damaged_objects
        )
        return [
            SortDamagedObjectsStage(
                damaged_objects=list(parameters.damaged_objects),
                target_zone=parameters.target_zone,
                minimum=minimum,
                instruction=(
                    parameters.instruction
                    if minimum == correctly_placed + 1
                    else "none"
                ),
                last=minimum == len(parameters.damaged_objects),
            )
            for minimum in range(
                correctly_placed + 1,
                len(parameters.damaged_objects) + 1,
            )
        ]

    def apply_request(
        self,
        state: TaskState,
        parameters: DamagedSortingParameters,
    ) -> TaskState:
        set_damaged_assignment(state, thaw_assignment(parameters.target))
        state.properties[LAST_DAMAGED_TARGET_PROPERTY_KEY] = parameters.target_zone
        state.properties[OBJECT_AREAS_KEY] = dict(
            parameters.resulting_object_areas
        )
        return state

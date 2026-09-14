import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.utils.text_utils import join_with_and

from ..brs_attributes import OBJECT_TYPES, ZONES, sorting_objects
from ..brs_stages import (
    AskDamagedObjectNamesStage,
    AskTypeZoneStage,
    AskZoneContentsStage,
)
from .common import (
    OBJECT_AREAS_KEY,
    TypedAssignment,
    format_zone,
    get_current_assignment,
    get_damaged_objects,
)


@dataclass(frozen=True)
class SortingQueryParameters:
    kind: str
    question: str
    answer: str
    target: str
    object_areas: Tuple[Tuple[str, str], ...]
    damaged_objects: Tuple[str, ...]


def _fully_sorted_types(assignment: TypedAssignment) -> Dict[str, str]:
    sorted_types = {}
    for object_type in OBJECT_TYPES:
        matching_zones = [
            zone
            for zone in ZONES
            if assignment[zone].get(object_type, 0)
            == len(sorting_objects[object_type])
        ]
        if len(matching_zones) == 1:
            sorted_types[object_type] = matching_zones[0]
    return sorted_types


def _zone_content_description(assignment: TypedAssignment, zone: str) -> str:
    descriptions = []
    for object_type in OBJECT_TYPES:
        quantity = assignment[zone].get(object_type, 0)
        if quantity:
            descriptions.append(
                f"{quantity} {object_type} object"
                if quantity == 1
                else f"{quantity} {object_type} objects"
            )
    readable_zone = format_zone(zone)
    if not descriptions:
        return f"The {readable_zone} is empty."
    return f"The {readable_zone} contains {join_with_and(descriptions)}."


class AskTypeZoneRequest(BaseRequest[SortingQueryParameters]):
    def __init__(self, weight: float = 0.7) -> None:
        super().__init__()
        self.weight = weight

    def sampling_weight(self, state: TaskState) -> float:
        return self.weight if _fully_sorted_types(get_current_assignment(state)) else 0

    def sample_parameters(self, state: TaskState) -> SortingQueryParameters:
        candidates = _fully_sorted_types(get_current_assignment(state))
        if not candidates:
            raise RuntimeError("No fruit type is completely grouped in one zone.")
        object_type = random.choice(list(candidates))
        zone = candidates[object_type]
        return SortingQueryParameters(
            kind="type_zone",
            question=f"In which tray are all {object_type} objects located?",
            answer=(
                f"All {object_type} objects are in the {format_zone(zone)}."
            ),
            target=object_type,
            object_areas=tuple(
                state.properties[OBJECT_AREAS_KEY].items()
            ),
            damaged_objects=tuple(get_damaged_objects(state)),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: SortingQueryParameters,
    ) -> List[BaseTaskStage]:
        zone = parameters.answer.removeprefix(
            f"All {parameters.target} objects are in the "
        ).removesuffix(".")
        return [AskTypeZoneStage(object_type=parameters.target, zone=zone)]


class AskZoneContentsRequest(BaseRequest[SortingQueryParameters]):
    def __init__(self, weight: float = 0.7) -> None:
        super().__init__()
        self.weight = weight

    def sampling_weight(self, state: TaskState) -> float:
        return self.weight

    def sample_parameters(self, state: TaskState) -> SortingQueryParameters:
        assignment = get_current_assignment(state)
        zone = random.choice(ZONES)
        description = _zone_content_description(assignment, zone)
        return SortingQueryParameters(
            kind="zone_contents",
            question=(
                f"What objects are currently in the {format_zone(zone)}?"
            ),
            answer=description,
            target=zone,
            object_areas=tuple(
                state.properties[OBJECT_AREAS_KEY].items()
            ),
            damaged_objects=tuple(get_damaged_objects(state)),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: SortingQueryParameters,
    ) -> List[BaseTaskStage]:
        return [
            AskZoneContentsStage(
                zone=format_zone(parameters.target),
                content_description=parameters.answer,
            )
        ]


class AskDamagedObjectsRequest(BaseRequest[SortingQueryParameters]):
    def __init__(self, weight: float = 0.7) -> None:
        super().__init__()
        self.weight = weight

    def sampling_weight(self, state: TaskState) -> float:
        return self.weight

    def sample_parameters(self, state: TaskState) -> SortingQueryParameters:
        damaged_objects = get_damaged_objects(state)
        answer = (
            "The damaged objects are " + ", ".join(damaged_objects) + "."
            if damaged_objects
            else "There are no damaged objects."
        )
        return SortingQueryParameters(
            kind="damaged_objects",
            question="Which objects are damaged?",
            answer=answer,
            target="damaged objects",
            object_areas=tuple(
                state.properties[OBJECT_AREAS_KEY].items()
            ),
            damaged_objects=tuple(damaged_objects),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: SortingQueryParameters,
    ) -> List[BaseTaskStage]:
        return [
            AskDamagedObjectNamesStage(
                damaged_objects=list(parameters.damaged_objects)
            )
        ]

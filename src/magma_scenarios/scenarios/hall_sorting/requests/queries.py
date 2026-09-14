import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest

from ..hs_stages import (
    AskHallTypesStage,
    AskPriorityHallStage,
    AskTypeHallAssignmentStage,
)
from .common import PRIORITY_HALL_KEY, TYPE_HALL_RELATION


QUERY_WEIGHT = 0.7
PRIORITY_QUERY_WEIGHT = 0.5


@dataclass(frozen=True)
class HallQueryParameters:
    kind: str
    targets: Tuple[str, ...]
    relations: Tuple[Tuple[str, str], ...]
    question: str
    answer: str


class AskTypeHallAssignmentRequest(BaseRequest[HallQueryParameters]):
    def __init__(
        self,
        max_types: int = 2,
        sampling_weight: float = QUERY_WEIGHT,
    ) -> None:
        super().__init__()
        if max_types < 1:
            raise ValueError("max_types must be at least 1.")
        if sampling_weight < 0:
            raise ValueError("sampling_weight must be non-negative.")
        self.max_types = max_types
        self.weight = sampling_weight

    def sampling_weight(self, state: TaskState) -> float:
        return self.weight if state.relations.get(TYPE_HALL_RELATION) else 0

    def sample_parameters(self, state: TaskState) -> HallQueryParameters:
        relations: Dict[str, str] = state.relations.get(TYPE_HALL_RELATION, {})
        if not relations:
            raise RuntimeError("No type-to-hall relations are available.")
        relation_types = list(relations)
        selected_count = random.randint(
            1,
            min(self.max_types, len(relation_types)),
        )
        selected_types = random.sample(relation_types, selected_count)
        stage = AskTypeHallAssignmentStage(relations, selected_types)
        return HallQueryParameters(
            "type_hall",
            tuple(selected_types),
            tuple(relations.items()),
            stage.get_stage_input().instruction.get_content(),
            stage.answer,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: HallQueryParameters,
    ) -> List[BaseTaskStage]:
        return [
            AskTypeHallAssignmentStage(
                dict(parameters.relations),
                list(parameters.targets),
            )
        ]


class AskHallTypesRequest(BaseRequest[HallQueryParameters]):
    def __init__(self, sampling_weight: float = QUERY_WEIGHT) -> None:
        super().__init__()
        if sampling_weight < 0:
            raise ValueError("sampling_weight must be non-negative.")
        self.weight = sampling_weight

    def sampling_weight(self, state: TaskState) -> float:
        return (
            self.weight
            if state.relations.get(TYPE_HALL_RELATION)
            and state.attributes.get("storage_halls")
            else 0
        )

    def sample_parameters(self, state: TaskState) -> HallQueryParameters:
        relations: Dict[str, str] = state.relations.get(TYPE_HALL_RELATION, {})
        storage_halls: List[str] = state.attributes.get("storage_halls", [])
        if not relations or not storage_halls:
            raise RuntimeError("Hall assignments are not available.")
        assigned_halls = set(relations.values())
        candidates = [hall for hall in storage_halls if hall in assigned_halls]
        if not candidates:
            raise RuntimeError("No storage hall currently has an assigned object type.")
        selected_hall = random.choice(candidates)
        stage = AskHallTypesStage(relations, selected_hall)
        return HallQueryParameters(
            "hall_types",
            (selected_hall,),
            tuple(relations.items()),
            stage.get_stage_input().instruction.get_content(),
            stage.answer,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: HallQueryParameters,
    ) -> List[BaseTaskStage]:
        return [
            AskHallTypesStage(
                dict(parameters.relations),
                parameters.targets[0],
            )
        ]


class AskPriorityHallRequest(BaseRequest[HallQueryParameters]):
    def __init__(self, sampling_weight: float = PRIORITY_QUERY_WEIGHT) -> None:
        super().__init__()
        if sampling_weight < 0:
            raise ValueError("sampling_weight must be non-negative.")
        self.weight = sampling_weight

    def sampling_weight(self, state: TaskState) -> float:
        return self.weight if state.properties.get(PRIORITY_HALL_KEY) else 0

    def sample_parameters(self, state: TaskState) -> HallQueryParameters:
        priority_hall = state.properties.get(PRIORITY_HALL_KEY)
        if priority_hall is None:
            raise RuntimeError("No hall-priority rule is currently active.")
        stage = AskPriorityHallStage(priority_hall)
        return HallQueryParameters(
            "priority_hall",
            (priority_hall,),
            (),
            stage.get_stage_input().instruction.get_content(),
            stage.answer,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: HallQueryParameters,
    ) -> List[BaseTaskStage]:
        return [AskPriorityHallStage(parameters.targets[0])]

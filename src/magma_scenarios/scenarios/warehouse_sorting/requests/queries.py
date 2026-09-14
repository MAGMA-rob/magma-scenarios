from dataclasses import dataclass
import random
from typing import Tuple

from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest

from ..warehouse_stages import (
    AskObjectAreaAssignementStage,
    AskObjectAreaAssignementStageInverse,
    inverse_object_area_query,
    object_area_query,
)


@dataclass(frozen=True)
class AreaObjectsQueryParameters:
    relations: Tuple[Tuple[str, str], ...]
    target_area: str

    @property
    def question(self) -> str:
        return object_area_query(dict(self.relations), self.target_area)[0]

    @property
    def answer(self) -> str:
        return object_area_query(dict(self.relations), self.target_area)[1]


@dataclass(frozen=True)
class ObjectAreasQueryParameters:
    relations: Tuple[Tuple[str, str], ...]
    objects: Tuple[str, ...]

    @property
    def question(self) -> str:
        return inverse_object_area_query(
            dict(self.relations),
            list(self.objects),
        )[0]

    @property
    def answer(self) -> str:
        return inverse_object_area_query(
            dict(self.relations),
            list(self.objects),
        )[1]


class AskObjectAreaAssignementRequest(BaseRequest[AreaObjectsQueryParameters]):
    def sampling_weight(self, state: TaskState) -> float:
        return 0.7 if state.relations.get("object_area", {}) else 0

    def sample_parameters(self, state: TaskState) -> AreaObjectsQueryParameters:
        relations = dict(state.relations.get("object_area", {}))
        if not relations:
            raise RuntimeError("No object-area relations available")
        selected_object = random.choice(list(relations))
        target_area = relations[selected_object]
        return AreaObjectsQueryParameters(
            tuple(relations.items()),
            target_area,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: AreaObjectsQueryParameters,
    ) -> list[BaseTaskStage]:
        return [AskObjectAreaAssignementStage(
            dict(parameters.relations),
            parameters.target_area,
        )]


class AskObjectAreaAssignementRequestInverse(
    BaseRequest[ObjectAreasQueryParameters]
):
    def __init__(self, max_objects: int = 2) -> None:
        super().__init__()
        self.max_objects = max_objects

    def sampling_weight(self, state: TaskState) -> float:
        return 0.7 if state.relations.get("object_area", {}) else 0

    def sample_parameters(self, state: TaskState) -> ObjectAreasQueryParameters:
        relations = dict(state.relations.get("object_area", {}))
        if not relations:
            raise RuntimeError("No object-area relations available")
        selected_objects = tuple(random.sample(
            list(relations),
            k=min(self.max_objects, len(relations)),
        ))
        return ObjectAreasQueryParameters(
            tuple(relations.items()),
            selected_objects,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: ObjectAreasQueryParameters,
    ) -> list[BaseTaskStage]:
        return [AskObjectAreaAssignementStageInverse(
            dict(parameters.relations),
            list(parameters.objects),
        )]

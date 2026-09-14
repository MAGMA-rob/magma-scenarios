import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest

from ..ad_stages import AskPendingReturnsCountStage, AskStorageProductCountStage
from ..attributes import PRODUCT_TYPES
from .common import OBJECT_LOCATION_RELATION


RECEPTION_LOCATION = "reception_grid"


def _storage_instances(state: TaskState, product_type: str) -> List[str]:
    locations: Dict[str, str] = state.relations.get(OBJECT_LOCATION_RELATION, {})
    target_grid = f"{product_type}_grid"
    return [name for name, location in locations.items() if location == target_grid]


def _reception_objects(state: TaskState) -> List[str]:
    locations: Dict[str, str] = state.relations.get(OBJECT_LOCATION_RELATION, {})
    return [
        name for name, location in locations.items()
        if location == RECEPTION_LOCATION
    ]


@dataclass(frozen=True)
class StorageProductCountParameters:
    product_type: str
    object_names: Tuple[str, ...]


@dataclass(frozen=True)
class PendingReturnsCountParameters:
    object_names: Tuple[str, ...]


class AskStorageProductCountRequest(
    BaseRequest[StorageProductCountParameters]
):
    def __init__(self, weight: float = 0.7) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError("weight must be non-negative.")
        self.weight = weight

    def _available_types(self, state: TaskState) -> List[str]:
        return [
            product_type
            for product_type in PRODUCT_TYPES
            if _storage_instances(state, product_type)
        ]

    def sampling_weight(self, state: TaskState) -> float:
        return self.weight if self._available_types(state) else 0

    def sample_parameters(
        self,
        state: TaskState,
    ) -> StorageProductCountParameters:
        available_types = self._available_types(state)
        if not available_types:
            raise RuntimeError("No product instances are currently in storage.")
        product_type = random.choice(available_types)
        instances = _storage_instances(state, product_type)
        return StorageProductCountParameters(
            product_type,
            tuple(instances),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: StorageProductCountParameters,
    ) -> List[BaseTaskStage]:
        return [
            AskStorageProductCountStage(
                parameters.product_type,
                list(parameters.object_names),
            )
        ]


class AskPendingReturnsCountRequest(
    BaseRequest[PendingReturnsCountParameters]
):
    def __init__(self, weight: float = 0.7) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError("weight must be non-negative.")
        self.weight = weight

    def sampling_weight(self, state: TaskState) -> float:
        return self.weight if _reception_objects(state) else 0

    def sample_parameters(
        self,
        state: TaskState,
    ) -> PendingReturnsCountParameters:
        objects = _reception_objects(state)
        if not objects:
            raise RuntimeError(
                "No returned products are currently waiting in reception."
            )
        return PendingReturnsCountParameters(tuple(objects))

    def create_stages(
        self,
        state: TaskState,
        parameters: PendingReturnsCountParameters,
    ) -> List[BaseTaskStage]:
        return [AskPendingReturnsCountStage(list(parameters.object_names))]

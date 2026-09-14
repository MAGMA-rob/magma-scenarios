from dataclasses import dataclass
import random
from typing import Dict, List, Tuple

from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.utils.text_utils import join_with_and

from ..laundry_constraints import CLOTHE_DETERGENT_KEY
from ..laundry_stages import (
    AskClothesDetergentStage,
    AskClothesDetergentStageInverse,
)
from .common import _get_known_clothes


@dataclass(frozen=True)
class DetergentQueryParameters:
    """Facts needed to answer which clothes use one detergent."""

    relations: Tuple[Tuple[str, str], ...]
    target_detergent: str
    clothes: Tuple[str, ...]

    @property
    def instruction(self) -> str:
        return f"Which clothes use '{self.target_detergent}'?"

    @property
    def answer(self) -> str:
        if not self.clothes:
            return f"No clothes use {self.target_detergent}."
        return (
            f"Clothes using '{self.target_detergent}' are: "
            f"{', '.join(self.clothes)}"
        )


@dataclass(frozen=True)
class ClothesQueryParameters:
    """Facts needed to answer which detergent applies to sampled clothes."""

    relations: Tuple[Tuple[str, str], ...]
    clothes: Tuple[str, ...]

    @property
    def instruction(self) -> str:
        noun = "detergents" if len(self.clothes) > 1 else "detergent"
        return f"What {noun} can wash {join_with_and(list(self.clothes))}?"

    @property
    def answer(self) -> str:
        grouped: Dict[str, List[str]] = {}
        relations = dict(self.relations)
        for clothe in self.clothes:
            grouped.setdefault(relations[clothe], []).append(clothe)
        return ", ".join(
            f"{join_with_and(group)} can be washed with {detergent}"
            for detergent, group in grouped.items()
        )


class AskClothesDetergentRequest(BaseRequest[DetergentQueryParameters]):
    def __init__(self, max_clothes: int = 3) -> None:
        super().__init__()
        self.max_clothes = max_clothes

    def sampling_weight(self, state: TaskState) -> float:
        return 0.6 if _get_known_clothes(state) else 0

    def sample_parameters(self, state: TaskState) -> DetergentQueryParameters:
        relations: Dict[str, str] = state.relations.get(CLOTHE_DETERGENT_KEY, {})
        if not relations:
            raise RuntimeError("No clothe-detergent relations available")
        target_detergent = random.choice(list(set(relations.values())))
        clothes = [
            clothe
            for clothe, detergent in relations.items()
            if detergent == target_detergent
        ]
        return DetergentQueryParameters(
            tuple(relations.items()),
            target_detergent,
            tuple(clothes),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: DetergentQueryParameters,
    ) -> List[BaseTaskStage]:
        return [
            AskClothesDetergentStage(
                dict(parameters.relations),
                parameters.target_detergent,
            )
        ]


class AskClothesDetergentRequestInverse(BaseRequest[ClothesQueryParameters]):
    def __init__(self, max_clothes: int = 3) -> None:
        super().__init__()
        self.max_clothes = max_clothes

    def sampling_weight(self, state: TaskState) -> float:
        return 0.6 if _get_known_clothes(state) else 0

    def sample_parameters(self, state: TaskState) -> ClothesQueryParameters:
        relations: Dict[str, str] = state.relations.get(CLOTHE_DETERGENT_KEY, {})
        if not relations:
            raise RuntimeError("No clothes-detergent relations available")
        clothes = random.sample(
            list(relations.keys()),
            k=min(self.max_clothes, len(relations)),
        )
        return ClothesQueryParameters(
            tuple(relations.items()),
            tuple(clothes),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: ClothesQueryParameters,
    ) -> List[BaseTaskStage]:
        return [
            AskClothesDetergentStageInverse(
                dict(parameters.relations),
                list(parameters.clothes),
            )
        ]

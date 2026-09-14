from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Sequence, Tuple

from magma_core.simulation.data_structures import EmptyInstruction, UserInstruction
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.utils.text_utils import join_with_and

from ..laundry_constraints import CLOTHE_DETERGENT_KEY
from ..laundry_stages import LoadClotheStage, WashStage


@dataclass(frozen=True)
class WashGroup:
    detergent: str
    clothes: Tuple[str, ...]


def _get_known_clothes(state: TaskState) -> List[str]:
    clothes = state.attributes.get("clothes", [])
    detergents = state.attributes.get("detergents", [])
    relations: Dict[str, str] = state.relations.get(CLOTHE_DETERGENT_KEY, {})
    return [
        clothe
        for clothe in clothes
        if clothe in relations and relations[clothe] in detergents
    ]


def _get_unassigned_clothes(state: TaskState) -> List[str]:
    clothes = state.attributes.get("clothes", [])
    relations: Dict[str, str] = state.relations.get(CLOTHE_DETERGENT_KEY, {})
    return [clothe for clothe in clothes if clothe not in relations]


def _sample_clothes(state: TaskState, max_clothes: int) -> List[str]:
    available_clothes = _get_known_clothes(state)
    if not available_clothes:
        raise RuntimeError("Failed to sample clothes with a known detergent")
    count = random.randint(1, min(max_clothes, len(available_clothes)))
    return random.sample(available_clothes, k=count)


def _group_clothes_by_detergent(
    state: TaskState,
    clothes: List[str],
) -> List[WashGroup]:
    relations: Dict[str, str] = state.relations.get(CLOTHE_DETERGENT_KEY, {})
    grouped: Dict[str, List[str]] = {}
    for clothe in clothes:
        grouped.setdefault(relations[clothe], []).append(clothe)
    return [
        WashGroup(detergent, tuple(group_clothes))
        for detergent, group_clothes in grouped.items()
    ]


def _build_all_groups_resolution_instruction(
    groups: Sequence[WashGroup],
) -> str:
    if len(groups) == 1:
        return f"You can use {groups[0].detergent}"
    washes = [
        f"wash {join_with_and(list(group.clothes))} with {group.detergent}"
        for group in groups
    ]
    return (
        f"Ok, do {len(groups)} washes in this exact order: first "
        + ", then ".join(washes)
        + "."
    )


def _build_wash_stages(
    detergent: str,
    clothes: Sequence[str],
    instruction: Optional[str] = None,
    link_first_stage_to_prev: bool = False,
    flag_answer: bool = True,
) -> List[BaseTaskStage]:
    normalized_clothes = list(clothes)
    first_instruction = (
        EmptyInstruction()
        if instruction is None
        else UserInstruction(instruction)
    )
    stages: List[BaseTaskStage] = [
        LoadClotheStage(1, normalized_clothes, first_instruction)
    ]
    if link_first_stage_to_prev:
        stages[0].stage_input.linked_to_prev = True
    for count in range(2, len(normalized_clothes) + 1):
        stages.append(LoadClotheStage(count, normalized_clothes))
    stages.append(
        WashStage(detergent, normalized_clothes, flag_answer=flag_answer)
    )
    return stages

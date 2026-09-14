from dataclasses import dataclass
import random
from typing import Dict, List, Optional

from magma_core.simulation.data_structures import UserInstruction
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.utils.text_utils import join_with_and
from magma_scenarios.templates.stages import MissingInformationStage

from ..laundry_constraints import CLOTHE_DETERGENT_KEY
from .common import (
    WashGroup,
    _build_all_groups_resolution_instruction,
    _build_wash_stages,
    _get_known_clothes,
    _get_unassigned_clothes,
    _group_clothes_by_detergent,
    _sample_clothes,
)


@dataclass(frozen=True)
class LaundryMissingInfo:
    """One clarification turn required before planning the wash."""

    question: str
    answer: str
    find_todo: str


@dataclass
class LaundryParameters:
    """The user turn and the physical washes selected by sampling."""

    user_instruction: str
    missing_info: Optional[LaundryMissingInfo]
    wash_groups: List[WashGroup]  # One group per physical wash.


class AskLaundryRequest(BaseRequest[LaundryParameters]):
    """Wash sampled clothes, clarifying only when detergent plans differ."""

    def __init__(self, max_clothes: int = 3) -> None:
        super().__init__()
        self.max_clothes = max_clothes

    def sampling_weight(self, state: TaskState) -> float:
        if not _get_known_clothes(state):
            return 0
        if state.properties.get(
            f"{CLOTHE_DETERGENT_KEY}_needs_application", False
        ):
            return 8
        relations: Dict[str, str] = state.relations.get(CLOTHE_DETERGENT_KEY, {})
        if len(set(relations.values())) > 1:
            return 4
        return 2

    def sample_parameters(self, state: TaskState) -> LaundryParameters:
        clothes = _sample_clothes(state, self.max_clothes)
        groups = _group_clothes_by_detergent(state, clothes)
        request_instruction = f"Please wash {join_with_and(clothes)}."
        if len(groups) == 1:
            return LaundryParameters(
                wash_groups=groups,
                user_instruction=request_instruction,
                missing_info=None
            )

        random.shuffle(groups)
        question = (
            "These clothes require different detergents. What should I do?"
        )
        if random.random() < 0.5:
            # One detergent for all
            selected_group = random.choice(groups)
            final_groups = [WashGroup(selected_group.detergent, tuple(clothes))]
            return LaundryParameters(
                wash_groups=final_groups,
                user_instruction=request_instruction,
                missing_info=LaundryMissingInfo(
                    question,
                    f"Ok, let's use {selected_group.detergent} for this time.",
                    "FIND whether to use one detergent or separate "
                    "detergent washes.",
                )
            )
        # different wash for each
        return LaundryParameters(
            wash_groups=groups,
            user_instruction=request_instruction,
            missing_info=LaundryMissingInfo(
                question,
                _build_all_groups_resolution_instruction(groups),
                "FIND whether to use one detergent or separate detergent "
                "washes.",
            ),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: LaundryParameters,
    ) -> List[BaseTaskStage]:
        stages: List[BaseTaskStage] = []
        if parameters.missing_info is not None:
            stages.append(
                MissingInformationStage(
                    UserInstruction(parameters.user_instruction),
                    "The robot must ask " + parameters.missing_info.question,
                )
            )
            execution_instruction = parameters.missing_info.answer
        else:
            execution_instruction = parameters.user_instruction
        for index, group in enumerate(parameters.wash_groups):
            stages.extend(
                _build_wash_stages(
                    group.detergent,
                    group.clothes,
                    (
                        execution_instruction
                        if index == 0
                        else None
                    ),
                    link_first_stage_to_prev=(
                        index == 0 and parameters.missing_info is not None
                    ),
                    flag_answer=index == len(parameters.wash_groups) - 1,
                )
            )
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: LaundryParameters,
    ) -> TaskState:
        key = f"{CLOTHE_DETERGENT_KEY}_needs_application"
        if state.properties.get(key, False):
            state.properties[key] = False
            applications_key = f"{CLOTHE_DETERGENT_KEY}_applications"
            state.properties[applications_key] = (
                state.properties.get(applications_key, 0) + 1
            )
        return state


class AskLaundryByDetergentRequest(AskLaundryRequest):
    """Wash known and unassigned clothes after asking for one detergent."""

    def sampling_weight(self, state: TaskState) -> float:
        if not _get_unassigned_clothes(state):
            return 0
        relations: Dict[str, str] = state.relations.get(CLOTHE_DETERGENT_KEY, {})
        return 5 if set(relations.values()) else 0

    def sample_parameters(self, state: TaskState) -> LaundryParameters:
        relations: Dict[str, str] = state.relations.get(CLOTHE_DETERGENT_KEY, {})
        unassigned_clothes = _get_unassigned_clothes(state)
        if not unassigned_clothes:
            raise RuntimeError("Failed to sample unassigned clothes")
        detergents = sorted(set(relations.values()))
        if not detergents:
            raise RuntimeError("Failed to sample a detergent")
        selected_detergent = random.choice(detergents)
        matching_clothes = [
            clothe
            for clothe, detergent in relations.items()
            if detergent == selected_detergent
        ]
        count = random.randint(
            1,
            min(len(unassigned_clothes), self.max_clothes),
        )
        selected_unassigned = random.sample(unassigned_clothes, k=count)
        remaining_slots = max(0, self.max_clothes - len(selected_unassigned))
        random.shuffle(matching_clothes)
        selected_matching = matching_clothes[:remaining_slots]
        selected_clothes = selected_matching + selected_unassigned
        request_instruction = f"Please wash {join_with_and(selected_clothes)}"
        joined_unassigned = join_with_and(selected_unassigned)
        return LaundryParameters(
            wash_groups=[WashGroup(selected_detergent, tuple(selected_clothes))],
            user_instruction=request_instruction,
            missing_info=LaundryMissingInfo(
                f"Which detergent should I use for {joined_unassigned}?",
                f"Use {selected_detergent} for this time.",
                f"FIND which detergent to use for {joined_unassigned}.",
            )
        )

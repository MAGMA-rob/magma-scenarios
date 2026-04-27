import random
from typing import Dict, List, Tuple

from magma_core.base.data_structures import EmptyInstruction, UserInstruction
from magma_core.base.stage import BaseTaskStage
from magma_core.base.state import TaskState
from magma_core.base.user_request import BaseRequest

from magma_scenarios.templates.requests import GiveRelationAssignmentRequest
from magma_scenarios.templates.stages import MissingInformationStage

from .laundry_stages import LoadClotheStage, WashStage

from .laundry_constraints import (
    CLOTHE_DETERGENT_KEY,
    ClotheDetergentConstraint,
)


class AssignClotheDetergentRequest(GiveRelationAssignmentRequest):
    """Sample permanent clothes-to-detergent compatibility rules."""

    def __init__(self, max_clothes_assignment: int = 9) -> None:
        super().__init__(
            relation_key=CLOTHE_DETERGENT_KEY,
            source_attribute_key="clothes",
            target_attribute_key="detergents",
            max_simultaneous_change=max_clothes_assignment,
            intro_message="Hello, please remember that ",
            assignment_template="{source} uses {target}",
            constraint_builder=ClotheDetergentConstraint,
            constraint_message_builder=_build_grouped_assignment_message,
        )

    def sampling_weight(self, state: TaskState) -> float:
        if state.properties.get(f"{CLOTHE_DETERGENT_KEY}_needs_application", False):
            return 0.25
        return super().sampling_weight(state)


def _join_clothes(clothes: List[str]) -> str:
    if len(clothes) == 1:
        return clothes[0]
    if len(clothes) == 2:
        return f"{clothes[0]} and {clothes[1]}"
    return ", ".join(clothes[:-1]) + f", and {clothes[-1]}"


def _build_grouped_assignment_message(intro_message: str, assignments: List[Tuple[str, str]]) -> str:
    grouped: Dict[str, List[str]] = {}
    for clothe, detergent in assignments:
        if detergent not in grouped:
            grouped[detergent] = []
        grouped[detergent].append(clothe)

    parts = [
        f"{_join_clothes(clothes)} use {detergent}"
        for detergent, clothes in grouped.items()
    ]
    return intro_message + ", ".join(parts) + "."


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
    if len(available_clothes) == 0:
        raise RuntimeError("Failed to sample clothes with a known detergent")

    nb_clothes = random.randint(1, min(max_clothes, len(available_clothes)))
    return random.sample(available_clothes, k=nb_clothes)


def _group_clothes_by_detergent(state: TaskState, clothes: List[str]) -> List[Tuple[str, List[str]]]:
    relations: Dict[str, str] = state.relations.get(CLOTHE_DETERGENT_KEY, {})
    grouped: Dict[str, List[str]] = {}

    for clothe in clothes:
        detergent = relations[clothe]
        if detergent not in grouped:
            grouped[detergent] = []
        grouped[detergent].append(clothe)

    return list(grouped.items())


def _build_single_wash_instruction(clothes: List[str]) -> str:
    return f"Please wash {_join_clothes(clothes)}."


def _build_multi_wash_instruction(groups: List[Tuple[str, List[str]]]) -> str:
    if len(groups) == 1:
        return _build_single_wash_instruction(groups[0][1])

    parts = []
    for detergent, clothes in groups:
        parts.append(f"wash {_join_clothes(clothes)} with {detergent}")

    if len(parts) == 2:
        sequence = " first " + parts[0] + ", then " + parts[1]
    else:
        sequence = " first " + parts[0]
        for part in parts[1:-1]:
            sequence += ", then " + part
        sequence += ", and then " + parts[-1]

    return f"Please do {len(groups)} washes in this exact order:{sequence}."


def _build_missing_information_answer(groups: List[Tuple[str, List[str]]]) -> str:
    parts = [f"{_join_clothes(clothes)} use {detergent}" for detergent, clothes in groups]
    if len(parts) == 2:
        detail = " and ".join(parts)
    else:
        detail = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return (
        "The model must ask the user what to do because "
        + detail
        + " and they require different detergents."
    )



def _build_all_groups_resolution_instruction(groups: List[Tuple[str, List[str]]]) -> str:
    if len(groups) == 1:
        detergent, clothes = groups[0]
        return f"You can use {detergent}"
    return "Ok, " + _build_multi_wash_instruction(groups)[7:]


def _build_wash_stages(
        detergent: str,
        clothes: List[str],
        instruction: str | None = None,
    ) -> List[BaseTaskStage]:
    stages: List[BaseTaskStage] = []
    first_instruction = EmptyInstruction() if instruction is None else UserInstruction(instruction)

    stages.append(LoadClotheStage(1, clothes, first_instruction))
    for count in range(2, len(clothes) + 1):
        stages.append(LoadClotheStage(count, clothes))
    stages.append(WashStage(detergent, clothes))

    return stages


class AskLaundryRequest(BaseRequest):
    """Ask to wash sampled clothes, with a clarification stage if detergents differ."""

    def __init__(self, max_clothes: int = 3) -> None:
        super().__init__()
        self.max_clothes = max_clothes

    def sampling_weight(self, state: TaskState) -> float:
        if len(_get_known_clothes(state)) == 0:
            return 0
        if state.properties.get(f"{CLOTHE_DETERGENT_KEY}_needs_application", False):
            return 8
        relations: Dict[str, str] = state.relations.get(CLOTHE_DETERGENT_KEY, {})
        if len(set(relations.values())) > 1:
            return 4
        return 2

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        clothes = _sample_clothes(state, self.max_clothes)
        groups = _group_clothes_by_detergent(state, clothes)
        request_instruction = _build_single_wash_instruction(clothes)

        if len(groups) == 1:
            detergent, compatible_clothes = groups[0]
            return _build_wash_stages(
                detergent,
                compatible_clothes,
                request_instruction,
            )

        stages: List[BaseTaskStage] = [
            MissingInformationStage(
                UserInstruction(request_instruction),
                _build_missing_information_answer(groups),
                state.memory,
                state.attributes,
            )
        ]

        random.shuffle(groups)
        if random.random() < 0.5:
            selected_detergent, selected_clothes = random.choice(groups)
            stages.extend(
                _build_wash_stages(
                    selected_detergent,
                    selected_clothes,
                    f"Ok, let's use {selected_detergent} for this time",
                )
            )
            return stages

        ordered_instruction = _build_all_groups_resolution_instruction(groups)
        for index, (detergent, compatible_clothes) in enumerate(groups):
            stages.extend(
                _build_wash_stages(
                    detergent,
                    compatible_clothes,
                    ordered_instruction if index == 0 else None,
                )
            )
            if index != len(groups)-1:
                stages[-1].situation.flag_answer_to_user = False
        return stages

    def apply_request(self, state: TaskState) -> TaskState:
        if state.properties.get(f"{CLOTHE_DETERGENT_KEY}_needs_application", False):
            state.properties[f"{CLOTHE_DETERGENT_KEY}_needs_application"] = False
            state.properties[f"{CLOTHE_DETERGENT_KEY}_applications"] = (
                state.properties.get(f"{CLOTHE_DETERGENT_KEY}_applications", 0) + 1
            )
        return state


class AskLaundryByDetergentRequest(BaseRequest):
    """Ask to wash all clothes for one detergent plus clothes with no assignment."""

    def __init__(self, max_clothes: int = 3) -> None:
        super().__init__()
        self.max_clothes = max_clothes

    def sampling_weight(self, state: TaskState) -> float:
        if len(_get_unassigned_clothes(state)) == 0:
            return 0

        relations: Dict[str, str] = state.relations.get(CLOTHE_DETERGENT_KEY, {})
        return 5 if len(set(relations.values())) > 0 else 0

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        relations: Dict[str, str] = state.relations.get(CLOTHE_DETERGENT_KEY, {})
        unassigned_clothes = _get_unassigned_clothes(state)
        if len(unassigned_clothes) == 0:
            raise RuntimeError("Failed to sample unassigned clothes")

        detergents = sorted(set(relations.values()))
        if len(detergents) == 0:
            raise RuntimeError("Failed to sample a detergent")

        selected_detergent = random.choice(detergents)
        matching_clothes = [
            clothe for clothe, detergent in relations.items()
            if detergent == selected_detergent
        ]

        nb_unassigned = random.randint(1, min(len(unassigned_clothes), self.max_clothes))
        selected_unassigned = random.sample(unassigned_clothes, k=nb_unassigned)

        remaining_slots = max(0, self.max_clothes - len(selected_unassigned))
        random.shuffle(matching_clothes)
        selected_matching = matching_clothes[:remaining_slots]

        selected_clothes = selected_matching + selected_unassigned

        request_instruction = f"Please wash {_join_clothes(selected_clothes)}"

        joined_clothes = _join_clothes(selected_unassigned)
        answer = f"The model must inform that {joined_clothes} do not have any known detergent and ask the user what detergent to use."

        stages: List[BaseTaskStage] = [
            MissingInformationStage(
                UserInstruction(request_instruction),
                answer,
                state.memory,
                state.attributes,
            )
        ]
        stages.extend(
            _build_wash_stages(
                selected_detergent,
                selected_clothes,
                f"Use {selected_detergent} for this time",
            )
        )
        return stages

    def apply_request(self, state: TaskState) -> TaskState:
        if state.properties.get(f"{CLOTHE_DETERGENT_KEY}_needs_application", False):
            state.properties[f"{CLOTHE_DETERGENT_KEY}_needs_application"] = False
            state.properties[f"{CLOTHE_DETERGENT_KEY}_applications"] = (
                state.properties.get(f"{CLOTHE_DETERGENT_KEY}_applications", 0) + 1
            )
        return state


class AskDirectLaundryRequest(BaseRequest):
    """Directly ask to wash a compatible group of clothes with one detergent."""

    def __init__(self, max_clothes: int = 3) -> None:
        super().__init__()
        self.max_clothes = max_clothes

    def sampling_weight(self, state: TaskState) -> float:
        if state.properties.get(f"{CLOTHE_DETERGENT_KEY}_needs_application", False):
            return 0.25
        if len(state.relations.get(CLOTHE_DETERGENT_KEY, {})) > 0:
            return 0.75
        return 1

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        clothes = state.attributes.get("clothes", [])
        
        nb_clothes = random.randint(1, min(self.max_clothes, len(clothes)))

        selected_detergent = random.choice(state.attributes["detergents"])
        selected_clothes = random.sample(clothes, k=nb_clothes)

        return _build_wash_stages(
            selected_detergent,
            selected_clothes,
            f"Please wash {_join_clothes(selected_clothes)} with {selected_detergent}."
        )

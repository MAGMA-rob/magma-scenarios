
from dataclasses import dataclass
import random
from typing import List, Literal, Tuple

from magma_core.simulation.data_structures import EmptyInstruction, UserInstruction
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.utils.text_utils import join_with_and

from ..laundry_constraints import CLOTHE_DETERGENT_KEY
from ..laundry_stages import LoadClotheStage, WashStage


@dataclass(frozen=True)
class LaundryInterruptionParameters:
    """Sampled facts describing the wash and its interruption."""

    kind: Literal["add_clothes", "change_detergent"]
    initial_clothes: Tuple[str, ...]
    added_clothes: Tuple[str, ...]
    initial_detergent: str
    final_detergent: str
    interrupt_after: int

    @property
    def final_clothes(self) -> Tuple[str, ...]:
        return self.initial_clothes + self.added_clothes

    @property
    def initial_instruction(self) -> str:
        return (
            f"Please wash {join_with_and(list(self.initial_clothes))} "
            f"with {self.initial_detergent}."
        )

    @property
    def interruption_instruction(self) -> str:
        if self.kind == "add_clothes":
            return (
                "I forgot some clothes: please also add "
                f"{join_with_and(list(self.added_clothes))} to this wash "
                f"and continue with {self.initial_detergent}."
            )
        return (
            f"I made a mistake: use {self.final_detergent} instead of "
            f"{self.initial_detergent} for this wash."
        )


class LaundryInterruptionRequest(BaseRequest[LaundryInterruptionParameters]):
    """Change an active laundry request while clothes are being loaded."""

    def __init__(
        self,
        max_initial_clothes: int = 3,
        max_added_clothes: int = 2,
    ) -> None:
        super().__init__()
        if max_initial_clothes < 2:
            raise ValueError("max_initial_clothes must be at least 2")
        if max_added_clothes < 1:
            raise ValueError("max_added_clothes must be at least 1")
        self.max_initial_clothes = max_initial_clothes
        self.max_added_clothes = max_added_clothes

    def sampling_weight(self, state: TaskState) -> float:
        clothes = state.attributes.get("clothes", [])
        detergents = state.attributes.get("detergents", [])
        if len(clothes) < 3 or not detergents:
            return 0
        if state.properties.get(
            f"{CLOTHE_DETERGENT_KEY}_needs_application", False
        ):
            return 0.25
        if state.relations.get(CLOTHE_DETERGENT_KEY, {}):
            return 0.75
        return 1

    def sample_parameters(
        self,
        state: TaskState,
    ) -> LaundryInterruptionParameters:
        all_clothes = state.attributes.get("clothes", []).copy()
        detergents = state.attributes.get("detergents", []).copy()
        if len(all_clothes) < 3:
            raise RuntimeError("At least three clothes are required.")
        if not detergents:
            raise RuntimeError("At least one detergent is required.")
        kinds = ["add_clothes"]
        if len(detergents) >= 2:
            kinds.append("change_detergent")
        kind = random.choice(kinds)
        random.shuffle(all_clothes)
        initial_limit = (
            len(all_clothes) - 1
            if kind == "add_clothes"
            else len(all_clothes)
        )
        initial_count = random.randint(
            2,
            min(self.max_initial_clothes, initial_limit),
        )
        initial_clothes = all_clothes[:initial_count]
        initial_detergent = random.choice(detergents)
        interrupt_after = random.randint(1, initial_count - 1)
        if kind == "add_clothes":
            remaining_clothes = all_clothes[initial_count:]
            added_count = random.randint(
                1,
                min(self.max_added_clothes, len(remaining_clothes)),
            )
            added_clothes = remaining_clothes[:added_count]
            final_detergent = initial_detergent
        else:
            added_clothes = []
            final_detergent = random.choice([
                detergent
                for detergent in detergents
                if detergent != initial_detergent
            ])
        return LaundryInterruptionParameters(
            kind=kind,
            initial_clothes=tuple(initial_clothes),
            added_clothes=tuple(added_clothes),
            initial_detergent=initial_detergent,
            final_detergent=final_detergent,
            interrupt_after=interrupt_after,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: LaundryInterruptionParameters,
    ) -> List[BaseTaskStage]:
        initial_clothes = list(parameters.initial_clothes)
        final_clothes = list(parameters.final_clothes)
        stages = [
            LoadClotheStage(
                count,
                initial_clothes,
                UserInstruction(parameters.initial_instruction)
                if count == 1
                else EmptyInstruction(),
            )
            for count in range(1, parameters.interrupt_after + 1)
        ]
        for count in range(
            parameters.interrupt_after + 1,
            len(final_clothes) + 1,
        ):
            stage = LoadClotheStage(
                count,
                final_clothes,
                UserInstruction(parameters.interruption_instruction)
                if count == parameters.interrupt_after + 1
                else EmptyInstruction(),
            )
            if count == parameters.interrupt_after + 1:
                stage.stage_input.linked_to_prev = True
            stages.append(stage)
        stages.append(
            WashStage(parameters.final_detergent, final_clothes)
        )
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: LaundryInterruptionParameters,
    ) -> TaskState:
        key = f"{CLOTHE_DETERGENT_KEY}_needs_application"
        if state.properties.get(key, False):
            state.properties[key] = False
            applications_key = f"{CLOTHE_DETERGENT_KEY}_applications"
            state.properties[applications_key] = (
                state.properties.get(applications_key, 0) + 1
            )
        return state

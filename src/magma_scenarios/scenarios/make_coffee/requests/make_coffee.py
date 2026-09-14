from dataclasses import dataclass
import random
from typing import List, Tuple

from magma_core.simulation.data_structures import UserInstruction
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.utils.text_utils import join_with_and
from magma_scenarios.templates.stages import ForbiddenElemStage

from ..coffee_constraints import (
    get_available_coffee_pods,
    get_unavailable_coffee_pods,
)
from ..coffee_stages import MakeOneCoffeStage
from .common import unavailable_coffee_verification


@dataclass(frozen=True)
class CoffeeRequestParameters:
    requested_pods: Tuple[str, ...]
    initial_instruction: str
    blocked_pods: Tuple[str, ...]
    served_pods: Tuple[str, ...]
    execution_instruction: str
    terminal_refusal: bool


class AskCoffeeRequest(BaseRequest[CoffeeRequestParameters]):
    def __init__(
        self,
        max_coffee_making: int = 4,
        strict_order: bool = True,
    ) -> None:
        super().__init__()
        self.max_nb = max_coffee_making
        self.strict_order = strict_order

    def sampling_weight(self, state: TaskState) -> float:
        if state.properties.get("coffee_preference_needs_application", False):
            return 0.25
        if state.properties.get(
            "team_coffee_preference_needs_application", False
        ):
            return 0.25
        return 0.75

    def sample_parameters(self, state: TaskState) -> CoffeeRequestParameters:
        pods = state.attributes.get("coffee_pod", [])
        if not pods:
            raise RuntimeError("Failed to find pods")
        requested_pods = random.choices(
            pods,
            k=random.randint(1, self.max_nb),
        )
        ordering = " in this exact order" if self.strict_order else ""
        if len(requested_pods) == 1:
            initial_instruction = f"Hey, serve me a {requested_pods[0]} coffee!"
        else:
            initial_instruction = (
                f"Hello, please make these coffees{ordering}: "
                + " and ".join(requested_pods)
            )

        unavailable = set(get_unavailable_coffee_pods(state))
        blocked_pods = list(dict.fromkeys(
            pod for pod in requested_pods if pod in unavailable
        ))
        if not blocked_pods:
            return CoffeeRequestParameters(
                tuple(requested_pods),
                initial_instruction,
                (),
                tuple(requested_pods),
                initial_instruction,
                False,
            )

        available_pods = get_available_coffee_pods(state)
        if len(requested_pods) == 1 and not available_pods:
            return CoffeeRequestParameters(
                tuple(requested_pods),
                initial_instruction,
                tuple(blocked_pods),
                (),
                "",
                True,
            )

        valid_pods = [pod for pod in requested_pods if pod not in unavailable]
        if valid_pods and random.choice([True, False]):
            served_pods = valid_pods
            execution_instruction = (
                f"Do not make the {join_with_and(blocked_pods)} coffee. "
                f"Please make only these coffees{ordering}: "
                f"{' and '.join(served_pods)}."
            )
        else:
            replacement_pod = random.choice(available_pods)
            served_pods = [
                replacement_pod if pod in unavailable else pod
                for pod in requested_pods
            ]
            execution_instruction = (
                f"Use {replacement_pod} coffee instead of "
                f"{join_with_and(blocked_pods)}. Please make these coffees "
                f"{'in this exact order: ' if self.strict_order else ''}"
                f"{' and '.join(served_pods)}."
            )
        return CoffeeRequestParameters(
            tuple(requested_pods),
            initial_instruction,
            tuple(blocked_pods),
            tuple(served_pods),
            execution_instruction,
            False,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: CoffeeRequestParameters,
    ) -> List[BaseTaskStage]:
        stages: List[BaseTaskStage] = []
        if parameters.blocked_pods:
            stages.append(
                ForbiddenElemStage(
                    UserInstruction(parameters.initial_instruction),
                    unavailable_coffee_verification(parameters.blocked_pods),
                )
            )
        if parameters.terminal_refusal:
            return stages
        linked_to_previous = bool(stages)
        for index, pod in enumerate(parameters.served_pods):
            stage = MakeOneCoffeStage(
                pod,
                parameters.execution_instruction if index == 0 else "none",
                flag_answer=index == len(parameters.served_pods) - 1,
            )
            if linked_to_previous and index == 0:
                stage.stage_input.linked_to_prev = True
            stages.append(stage)
        return stages


from dataclasses import dataclass
import random
from typing import Dict, List, Tuple

from magma_core.simulation.data_structures import UserInstruction
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.utils.text_utils import join_with_and
from magma_scenarios.templates.stages import (
    ForbiddenElemStage,
    MissingInformationStage,
)

from ..coffee_constraints import (
    get_available_coffee_pods,
    get_unavailable_coffee_pods,
)
from ..coffee_stages import MakeOneCoffeStage
from .common import (
    cancel_coffee_instruction,
    missing_preference_verification,
    ordered_coffee_instruction,
    preference_resolution,
    substitute_coffee_instruction,
    unavailable_people_verification,
)


@dataclass(frozen=True)
class PerUserCoffeeParameters:
    names: Tuple[str, ...]
    missing_preferences: Tuple[Tuple[str, str], ...]
    capsule_by_name: Tuple[Tuple[str, str], ...]
    request_instruction: str
    ordered_instruction: str
    resolution_instruction: str
    blocked_names: Tuple[str, ...]
    served_names: Tuple[str, ...]
    served_capsules: Tuple[Tuple[str, str], ...]
    execution_instruction: str
    terminal_refusal: bool


class AskCoffeePerUser(BaseRequest[PerUserCoffeeParameters]):
    def __init__(
        self,
        possible_names: List[str],
        max_coffee: int = 3,
        force_order: bool = True,
    ) -> None:
        super().__init__()
        self.nb = max_coffee
        self.possible_names = possible_names
        self.force_order = force_order

    def sampling_weight(self, state: TaskState) -> float:
        if len(state.attributes.get("coffee_pod", [])) == 0:
            return 0
        if state.properties.get("coffee_preference_needs_application", False):
            return 8
        if state.properties.get(
            "team_coffee_preference_needs_application", False
        ):
            return 6
        if len(state.relations.get("coffee_preference", {})) > 0:
            return 4
        return 2

    def _sample_requested_users(
        self,
        state: TaskState,
    ) -> tuple[List[str], Dict[str, str]]:
        pods = state.attributes.get("coffee_pod", [])
        if not pods:
            raise RuntimeError("Empty coffee pod")
        preferences: Dict[str, str] = state.relations.get(
            "coffee_preference", {}
        )
        known_names = [
            name for name in self.possible_names if name in preferences
        ]
        random.shuffle(known_names)
        all_names = list(known_names)
        all_names.extend(
            name
            for name in self.possible_names
            if name not in preferences and name not in all_names
        )
        if not all_names:
            raise RuntimeError("Failed to sample a user to serve coffee")
        count = random.randint(1, min(self.nb, len(all_names)))
        selected_names = known_names[:count]
        if len(selected_names) < count:
            missing_names = [
                name for name in all_names if name not in preferences
            ]
            random.shuffle(missing_names)
            selected_names.extend(missing_names[:count - len(selected_names)])
        missing_preferences = {
            name: random.choice(pods)
            for name in selected_names
            if name not in preferences
        }
        return selected_names, missing_preferences

    def sample_parameters(self, state: TaskState) -> PerUserCoffeeParameters:
        preferences: Dict[str, str] = state.relations.get(
            "coffee_preference", {}
        )
        names, missing_preferences = self._sample_requested_users(state)
        request_instruction = f"Please make coffee for {join_with_and(names)}."
        ordered_instruction = (
            ordered_coffee_instruction(names)
            if self.force_order
            else request_instruction
        )
        resolution_instruction = preference_resolution(missing_preferences)
        capsule_by_name = {
            name: missing_preferences.get(name, preferences.get(name))
            for name in names
        }
        if any(pod is None for pod in capsule_by_name.values()):
            raise RuntimeError("A sampled coffee preference is missing.")
        unavailable = set(get_unavailable_coffee_pods(state))
        blocked_names = [
            name
            for name in names
            if capsule_by_name[name] in unavailable
        ]
        first_instruction = (
            f"{resolution_instruction} {ordered_instruction}"
            if resolution_instruction
            else ordered_instruction
        )
        if blocked_names:
            available_pods = get_available_coffee_pods(state)
            terminal_refusal = len(names) == 1 or not available_pods
            if terminal_refusal:
                return PerUserCoffeeParameters(
                    tuple(names),
                    tuple(missing_preferences.items()),
                    tuple(capsule_by_name.items()),
                    request_instruction,
                    ordered_instruction,
                    resolution_instruction,
                    tuple(blocked_names),
                    (),
                    (),
                    first_instruction,
                    True,
                )
            valid_names = [name for name in names if name not in blocked_names]
            should_cancel = bool(valid_names) and random.choice([True, False])
            if should_cancel:
                served_names = valid_names
                served_capsules = capsule_by_name.copy()
                execution_instruction = cancel_coffee_instruction(
                    blocked_names,
                    served_names,
                )
            else:
                served_names = names
                replacement_pod = random.choice(available_pods)
                served_capsules = capsule_by_name.copy()
                for name in blocked_names:
                    served_capsules[name] = replacement_pod
                execution_instruction = substitute_coffee_instruction(
                    blocked_names,
                    replacement_pod,
                    served_names,
                    ordered=self.force_order,
                )
            return PerUserCoffeeParameters(
                tuple(names),
                tuple(missing_preferences.items()),
                tuple(capsule_by_name.items()),
                request_instruction,
                ordered_instruction,
                resolution_instruction,
                tuple(blocked_names),
                tuple(served_names),
                tuple(served_capsules.items()),
                execution_instruction,
                False,
            )

        if not self.force_order:
            first_instruction = resolution_instruction or request_instruction
        return PerUserCoffeeParameters(
            tuple(names),
            tuple(missing_preferences.items()),
            tuple(capsule_by_name.items()),
            request_instruction,
            ordered_instruction,
            resolution_instruction,
            (),
            tuple(names),
            tuple(capsule_by_name.items()),
            first_instruction,
            False,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: PerUserCoffeeParameters,
    ) -> List[BaseTaskStage]:
        stages: List[BaseTaskStage] = []
        if parameters.missing_preferences:
            stages.append(
                MissingInformationStage(
                    UserInstruction(parameters.request_instruction),
                    missing_preference_verification(
                        [name for name, _ in parameters.missing_preferences]
                    ),
                )
            )
        capsule_by_name = dict(parameters.capsule_by_name)
        if parameters.blocked_names:
            refusal_stage = ForbiddenElemStage(
                UserInstruction(
                    parameters.execution_instruction
                    if parameters.terminal_refusal
                    else (
                        f"{parameters.resolution_instruction} {parameters.ordered_instruction}"
                        if parameters.resolution_instruction
                        else parameters.ordered_instruction
                    ),
                    has_constraint=bool(parameters.resolution_instruction),
                ),
                unavailable_people_verification(
                    parameters.blocked_names,
                    capsule_by_name,
                ),
            )
            if stages:
                refusal_stage.stage_input.linked_to_prev = True
            stages.append(refusal_stage)
        if parameters.terminal_refusal:
            return stages

        served_capsules = dict(parameters.served_capsules)
        link_first = bool(stages)
        for index, name in enumerate(parameters.served_names):
            stage = MakeOneCoffeStage(
                served_capsules[name],
                parameters.execution_instruction if index == 0 else "none",
                index == len(parameters.served_names) - 1,
            )
            if link_first and index == 0:
                stage.stage_input.linked_to_prev = True
            stages.append(stage)
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: PerUserCoffeeParameters,
    ) -> TaskState:
        if state.properties.get("coffee_preference_needs_application", False):
            state.properties["coffee_preference_needs_application"] = False
            state.properties["coffee_preference_applications"] = (
                state.properties.get("coffee_preference_applications", 0) + 1
            )
        return state

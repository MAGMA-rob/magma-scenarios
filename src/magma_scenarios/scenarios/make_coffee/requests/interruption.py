from dataclasses import dataclass
import random
from typing import List, Tuple

from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.utils.text_utils import join_with_and

from ..coffee_constraints import get_available_coffee_pods
from ..coffee_stages import MakeOneCoffeStage
from .common import CoffeeOrderItem


@dataclass(frozen=True)
class CoffeeInterruptionParameters:
    items: Tuple[CoffeeOrderItem, ...]
    kind: str
    visible_indices: Tuple[int, ...]
    added_indices: Tuple[int, ...]
    interruption_index: int
    initial_instruction: str
    interruption_instruction: str


class CoffeeInterruptionRequest(BaseRequest[CoffeeInterruptionParameters]):
    """Make an ordered coffee request and interrupt it once."""

    def __init__(self, possible_names: List[str], max_coffee: int = 4) -> None:
        super().__init__()
        if max_coffee < 3:
            raise ValueError("max_coffee must be at least 3")
        self.possible_names = possible_names
        self.max_coffee = max_coffee

    def sampling_weight(self, state: TaskState) -> float:
        if not get_available_coffee_pods(state):
            return 0
        if state.properties.get("coffee_preference_needs_application", False):
            return 0.25
        return 4

    @staticmethod
    def _describe_items(items: List[CoffeeOrderItem]) -> str:
        return join_with_and([
            f"a coffee for {name}"
            if name is not None
            else f"a {pod} coffee"
            for pod, name in items
        ])

    def sample_parameters(
        self,
        state: TaskState,
    ) -> CoffeeInterruptionParameters:
        available_pods = get_available_coffee_pods(state)
        if not available_pods:
            raise RuntimeError("An available coffee pod is required.")
        preferences = state.relations.get("coffee_preference", {})
        known_names = [
            name
            for name in self.possible_names
            if preferences.get(name) in available_pods
        ]
        coffee_count = random.randint(3, self.max_coffee)
        named_count = random.randint(0, min(len(known_names), coffee_count))
        selected_names = random.sample(known_names, k=named_count)
        items: List[CoffeeOrderItem] = [
            (preferences[name], name) for name in selected_names
        ]
        items.extend(
            (random.choice(available_pods), None)
            for _ in range(coffee_count - named_count)
        )
        random.shuffle(items)

        kind = random.choice(["append_later", "priority"])
        if kind == "append_later":
            visible_count = random.randint(2, coffee_count - 1)
            visible_indices = list(range(visible_count))
            added_indices = list(range(visible_count, coffee_count))
            interruption_index = random.randint(1, visible_count - 1)
            interruption_instruction = (
                "When you have finished the coffees already requested, please "
                f"also make {self._describe_items([items[index] for index in added_indices])}."
            )
        else:
            interruption_index = random.randint(1, coffee_count - 2)
            visible_indices = [
                index for index in range(coffee_count)
                if index != interruption_index
            ]
            added_indices = [interruption_index]
            interruption_instruction = (
                f"Do {self._describe_items([items[interruption_index]])} "
                "as a priority now, then resume the previous order."
            )
        initial_instruction = (
            "Please make these coffees in this exact order: "
            f"{self._describe_items([items[index] for index in visible_indices])}."
        )
        return CoffeeInterruptionParameters(
            tuple(items),
            kind,
            tuple(visible_indices),
            tuple(added_indices),
            interruption_index,
            initial_instruction,
            interruption_instruction,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: CoffeeInterruptionParameters,
    ) -> List[BaseTaskStage]:
        stages = []
        for index, (pod, _) in enumerate(parameters.items):
            stage = MakeOneCoffeStage(
                pod,
                parameters.initial_instruction
                if index == 0
                else parameters.interruption_instruction
                if index == parameters.interruption_index
                else "none",
                flag_answer=index == len(parameters.items) - 1,
            )
            if index == parameters.interruption_index:
                stage.stage_input.linked_to_prev = True
            stages.append(stage)
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: CoffeeInterruptionParameters,
    ) -> TaskState:
        if state.properties.get("coffee_preference_needs_application", False):
            state.properties["coffee_preference_needs_application"] = False
            state.properties["coffee_preference_applications"] = (
                state.properties.get("coffee_preference_applications", 0) + 1
            )
        return state

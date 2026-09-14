from collections import Counter
import random
from typing import Dict

from magma_core.simulation.constraints import BaseConstraint
from magma_core.simulation.state import TaskState
from magma_scenarios.templates.requests.interact_request import (
    BaseConstraintRequest,
    ConstraintParameters,
)

from ...common.planning import requirements_text
from .common import available_requirement_slots


class TableAssignmentConstraint(BaseConstraint):
    def __init__(self, requirements: Dict[str, int]) -> None:
        super().__init__()
        self.requirements = requirements.copy()

    def apply(self, state: TaskState):
        super().apply(state)
        state.relations["table_requirements"] = self.requirements.copy()
        state.properties["table_requirements_needs_application"] = True

    def outdated(self, state: TaskState) -> bool:
        available = Counter(
            available_requirement_slots(
                state.properties.get("active_objects", [])
            )
        )
        return any(
            available[requirement] < count
            for requirement, count in self.requirements.items()
        )


class GiveTableAssignmentRequest(BaseConstraintRequest):
    reset_at_end = False

    def __init__(
        self,
        max_table_objects: int = 4,
        empty_rule_weight: float = 4.0,
        existing_rule_weight: float = 1.0,
        requests_to_normal_weight: int = 4,
    ) -> None:
        super().__init__()
        if max_table_objects < 2:
            raise ValueError("max_table_objects must be superior at 1.")
        if min(empty_rule_weight, existing_rule_weight) < 0:
            raise ValueError("Sampling weights must be non-negative.")
        if requests_to_normal_weight <= 0:
            raise ValueError("requests_to_normal_weight must be positive.")
        self.max_table_objects = max_table_objects
        self.empty_rule_weight = empty_rule_weight
        self.existing_rule_weight = existing_rule_weight
        self.requests_to_normal_weight = requests_to_normal_weight

    def sampling_weight(self, state: TaskState) -> float:
        if not available_requirement_slots(
            state.properties.get("active_objects", [])
        ):
            return 0
        if not state.relations.get("table_requirements"):
            return self.empty_rule_weight

        request_index = state.properties.get("_generator_request_index")
        last_assignment_index = state.properties.get(
            "table_assignment_last_request_index"
        )
        if request_index is None or last_assignment_index is None:
            return self.existing_rule_weight

        intermediate_requests = max(
            0,
            request_index - last_assignment_index,
        )
        age_ratio = min(
            1.0,
            intermediate_requests / self.requests_to_normal_weight,
        )
        return self.existing_rule_weight * age_ratio

    def apply_request(
        self,
        state: TaskState,
        parameters: ConstraintParameters,
    ) -> TaskState:
        state = super().apply_request(state, parameters)
        request_index = state.properties.get("_generator_request_index")
        if request_index is not None:
            state.properties["table_assignment_last_request_index"] = (
                request_index + 1
            )
        return state

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        slots = available_requirement_slots(
            state.properties.get("active_objects", [])
        )
        if not slots:
            raise RuntimeError("No active object can be used to set the table.")

        count = random.randint(2, min(self.max_table_objects, len(slots)))
        sampled_requirements = dict(Counter(random.sample(slots, k=count)))
        current_requirements = state.relations.get("table_requirements", {})

        if sampled_requirements == current_requirements:
            alternatives = [
                requirement
                for requirement in set(slots)
                if requirement not in sampled_requirements
            ]
            if alternatives:
                removed = next(iter(sampled_requirements))
                sampled_requirements[removed] -= 1
                if sampled_requirements[removed] == 0:
                    del sampled_requirements[removed]
                replacement = random.choice(alternatives)
                sampled_requirements[replacement] = 1

        instruction = (
            "From now on, when I ask you to set the table, put "
            f"{requirements_text(sampled_requirements)} on it."
        )
        return ConstraintParameters(
            [TableAssignmentConstraint(sampled_requirements)],
            instruction,
        )

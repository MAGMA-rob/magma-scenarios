import random
from typing import List
from magma_core.simulation.constraints import BaseConstraint
from magma_core.simulation.state import TaskState
from magma_scenarios.templates.requests.interact_request import (
    BaseConstraintRequest,
    ConstraintParameters,
)
from ..brs_attributes import OBJECT_TYPES, ZONES
from .common import (
    TYPE_PRIORITY_KEY,
    TYPE_PRIORITY_NEEDS_APPLICATION_KEY,
    get_current_assignment,
)


PRIORITY_ACTIVATION_WEIGHT = 2.0
PRIORITY_FORGET_WEIGHT = 1.0


def _priority_type_candidates(state: TaskState) -> List[str]:
    """Return object types having at least one available instance."""

    assignment = get_current_assignment(state)

    return [
        object_type
        for object_type in OBJECT_TYPES
        if any(
            assignment[zone].get(object_type, 0) > 0
            for zone in ZONES
        )
    ]


class TypePriorityConstraint(BaseConstraint):
    """Require one fruit type to always be handled first."""

    def __init__(self, object_type: str) -> None:
        super().__init__()
        self.object_type = object_type

    def apply(self, state: TaskState) -> None:
        super().apply(state)

        state.properties[TYPE_PRIORITY_KEY] = self.object_type

        state.properties[TYPE_PRIORITY_NEEDS_APPLICATION_KEY] = True

    def outdated(self, state: TaskState) -> bool:
        return (
            state.properties.get(TYPE_PRIORITY_KEY)
            != self.object_type
        )


class ForgetTypePriorityConstraint(BaseConstraint):
    """Remove the currently active fruit-type priority."""

    def apply(self, state: TaskState) -> None:
        super().apply(state)

        state.properties.pop(TYPE_PRIORITY_KEY, None)

        state.properties.pop(TYPE_PRIORITY_NEEDS_APPLICATION_KEY, None)

    def outdated(self, state: TaskState) -> bool:
        return (
            state.properties.get(TYPE_PRIORITY_KEY)
            is not None
        )


class TypePriorityRequest(BaseConstraintRequest):
    """
    Activate a fruit-type priority or forget the active priority.

    No active priority:
        "Always sort all banana objects first."

    Existing applied priority:
        "Forget the rule requiring banana objects to be sorted first."
    """

    reset_at_end = False

    def __init__(
        self,
        activation_weight: float = PRIORITY_ACTIVATION_WEIGHT,
        forget_weight: float = PRIORITY_FORGET_WEIGHT,
    ) -> None:
        super().__init__()

        if activation_weight < 0:
            raise ValueError("activation_weight must be non-negative.")

        if forget_weight < 0:
            raise ValueError("forget_weight must be non-negative.")

        self.activation_weight = activation_weight
        self.forget_weight = forget_weight

    def sampling_weight(self, state: TaskState) -> float:
        # The active priority must be exercised before it can
        # be forgotten.
        if state.properties.get(TYPE_PRIORITY_NEEDS_APPLICATION_KEY, False):
            return 0

        active_priority = state.properties.get(TYPE_PRIORITY_KEY)

        if active_priority is not None:
            return self.forget_weight

        if not _priority_type_candidates(state):
            return 0

        return self.activation_weight

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        active_priority = state.properties.get(TYPE_PRIORITY_KEY)

        if active_priority is not None:
            instruction = (
                f"Forget the rule requiring all "
                f"{active_priority} objects to be sorted first."
            )
            return ConstraintParameters(
                [ForgetTypePriorityConstraint()],
                instruction,
            )

        candidates = _priority_type_candidates(state)

        if not candidates:
            raise RuntimeError("No fruit type can receive a priority rule.")

        selected_type = random.choice(candidates)

        instruction = (
            f"Each time I ask you to sort fruits, always "
            f"sort all {selected_type} objects first."
        )
        return ConstraintParameters(
            [TypePriorityConstraint(selected_type)],
            instruction,
        )

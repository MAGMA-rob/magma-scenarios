import random
from typing import List

from magma_core.simulation.constraints import BaseConstraint
from magma_core.simulation.state import TaskState
from magma_scenarios.templates.requests.interact_request import (
    BaseConstraintRequest,
    ConstraintParameters,
)

from .common import (
    PRIORITY_HALL_KEY,
    PRIORITY_HALL_NEEDS_APPLICATION_KEY,
    TYPE_HALL_RELATION,
)


PRIORITY_ACTIVATION_WEIGHT = 1.0
PRIORITY_FORGET_WEIGHT = 0.5


def _priority_hall_candidates(state: TaskState) -> List[str]:
    """
    Return storage halls that currently receive at least one object type.
    """
    storage_halls = state.attributes.get(
        "storage_halls",
        [],
    )

    assigned_halls = set(
        state.relations.get(
            TYPE_HALL_RELATION,
            {},
        ).values()
    )

    return [
        hall
        for hall in storage_halls
        if hall in assigned_halls
    ]


class HallPriorityConstraint(BaseConstraint):
    """Require one storage hall to be completed first."""

    reset_at_end = False

    def __init__(self, hall: str) -> None:
        super().__init__()
        self.hall = hall

    def apply(self, state: TaskState) -> None:
        super().apply(state)

        state.properties[PRIORITY_HALL_KEY] = self.hall

        state.properties[
            PRIORITY_HALL_NEEDS_APPLICATION_KEY
        ] = True

    def outdated(self, state: TaskState) -> bool:
        return (
            state.properties.get(PRIORITY_HALL_KEY)
            != self.hall
        )


class ForgetHallPriorityConstraint(BaseConstraint):
    """Remove the currently active hall-priority rule."""

    def apply(self, state: TaskState) -> None:
        super().apply(state)

        state.properties.pop(
            PRIORITY_HALL_KEY,
            None,
        )

        state.properties.pop(
            PRIORITY_HALL_NEEDS_APPLICATION_KEY,
            None,
        )

    def outdated(self, state: TaskState) -> bool:
        return (
            state.properties.get(PRIORITY_HALL_KEY)
            is not None
        )


class HallPriorityRequest(BaseConstraintRequest):
    """
    Activate a hall-priority rule or forget the current one.

    No active priority:
        "Always complete Hall3 first."

    Existing applied priority:
        "Forget the current Hall3 priority rule."
    """

    def __init__(
        self,
        activation_weight: float = PRIORITY_ACTIVATION_WEIGHT,
        forget_weight: float = PRIORITY_FORGET_WEIGHT,
    ) -> None:
        super().__init__()

        if activation_weight < 0:
            raise ValueError(
                "activation_weight must be non-negative."
            )

        if forget_weight < 0:
            raise ValueError(
                "forget_weight must be non-negative."
            )

        self.activation_weight = activation_weight
        self.forget_weight = forget_weight

    def sampling_weight(self, state: TaskState) -> float:
        # Do not forget or replace a priority before a delivery
        # has applied it.
        if state.properties.get(
            PRIORITY_HALL_NEEDS_APPLICATION_KEY,
            False,
        ):
            return 0

        active_priority = state.properties.get(
            PRIORITY_HALL_KEY
        )

        if active_priority is not None:
            return self.forget_weight

        if not _priority_hall_candidates(state):
            return 0

        return self.activation_weight

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        active_priority = state.properties.get(
            PRIORITY_HALL_KEY
        )

        if active_priority is not None:
            instruction = (
                f"{active_priority} no longer has priority."
            )
            return ConstraintParameters(
                [ForgetHallPriorityConstraint()],
                instruction,
            )

        candidates = _priority_hall_candidates(state)

        if not candidates:
            raise RuntimeError(
                "No storage hall can currently receive "
                "a priority rule."
            )

        selected_hall = random.choice(candidates)

        instruction = (
            f"For future deliveries, complete {selected_hall} first."
        )
        return ConstraintParameters(
            [HallPriorityConstraint(selected_hall)],
            instruction,
        )

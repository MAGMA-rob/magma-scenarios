import random

from magma_core.simulation.state import TaskState
from magma_scenarios.templates.requests.interact_request import (
    BaseConstraintRequest,
    ConstraintParameters,
)

from ..coffee_constraints import (
    CoffeeAvailableConstraint,
    CoffeeUnavailableConstraint,
    get_unavailable_coffee_pods,
)


class ToggleCoffeeAvailability(BaseConstraintRequest):
    """Toggle one persistent coffee pod unavailability."""

    def __init__(self, max_unavailable: int = 1) -> None:
        super().__init__()
        if max_unavailable != 1:
            raise ValueError("Only one unavailable coffee pod is supported for now")
        self.max_unavailable = max_unavailable

    def sampling_weight(self, state: TaskState) -> float:
        pods = state.attributes.get("coffee_pod", [])
        if len(pods) <= 0:
            return 0
        if get_unavailable_coffee_pods(state):
            return 0.75
        if len(pods) <= self.max_unavailable:
            return 0
        return 1

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        unavailable_pods = get_unavailable_coffee_pods(state)

        if unavailable_pods:
            coffee = unavailable_pods[0]
            templates = (
                f"{coffee} coffee is available again.",
                f"We have {coffee} capsules again now.",
                f"The {coffee} capsules have been restocked.",
                f"You can use {coffee} coffee again now.",
            )
            return ConstraintParameters(
                [CoffeeAvailableConstraint(coffee)],
                random.choice(templates),
            )

        pods = state.attributes.get("coffee_pod", [])
        if len(pods) <= self.max_unavailable:
            raise RuntimeError(
                f"Failed to build the stage from {self.__class__.__name__} "
                "because too few coffee pods are known"
            )

        coffee = random.choice(pods)
        templates = (
            f"{coffee} coffee is not available anymore.",
            f"There are no {coffee} capsules left for now.",
            f"We are out of {coffee} capsules until further notice.",
        )
        return ConstraintParameters(
            [CoffeeUnavailableConstraint(coffee)],
            random.choice(templates),
        )

import random

from magma_core.simulation.constraints import BaseConstraint
from magma_scenarios.templates.requests.interact_request import (
    BaseConstraintRequest,
    ConstraintParameters,
)
from magma_core.simulation.state import TaskState

from .common import get_colors


class FirstColorConstraint(BaseConstraint):
    def apply(self, state: TaskState) -> None:
        super().apply(state)
        state.properties["constraint_order"] = "first-color"
        state.properties["constraint_order_needs_application"] = True


class SecondColorConstraint(BaseConstraint):
    def apply(self, state: TaskState) -> None:
        super().apply(state)
        state.properties["constraint_order"] = "second-color"
        state.properties["constraint_order_needs_application"] = True


class ForgetOrderConstraint(BaseConstraint):
    def apply(self, state: TaskState) -> None:
        super().apply(state)
        state.properties.pop("constraint_order", None)
        state.properties.pop("constraint_order_needs_application", None)
        state.properties.pop("constraint_order_applications", None)


class GiveOrderConstraint(BaseConstraintRequest):
    def sampling_weight(self, state: TaskState) -> float:
        if state.properties.get("constraint_order_needs_application", False):
            return 0
        if state.properties.get("constraint_order") is None:
            return 8
        return 3

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        colors = get_colors(state)
        active_rule = state.properties.get("constraint_order")
        if active_rule == "first-color":
            return ConstraintParameters(
                [ForgetOrderConstraint()],
                f"Forget about always sorting {colors[0]} cubes first.",
            )
        if active_rule == "second-color":
            return ConstraintParameters(
                [ForgetOrderConstraint()],
                f"Forget about always sorting {colors[1]} cubes first.",
            )

        selected_rule = random.choice(("first-color", "second-color"))
        if selected_rule == "first-color":
            constraint = FirstColorConstraint()
            priority_color = colors[0]
        else:
            constraint = SecondColorConstraint()
            priority_color = colors[1]
        return ConstraintParameters(
            [constraint],
            "Each time I ask you to sort cubes, you must always sort all "
            f"{priority_color} cubes first.",
        )

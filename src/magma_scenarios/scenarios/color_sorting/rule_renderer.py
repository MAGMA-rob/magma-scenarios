from typing import List

from magma_core.simulation.state import RuleRenderer, TaskState


class ColorSortingRuleRenderer(RuleRenderer):
    """Project the active color-order rule from the scenario state."""

    def rules(self, state: TaskState) -> List[str]:
        rule = state.properties.get("constraint_order")
        if rule is None:
            return []
        colors = state.attributes.get("known_tray_color", [])
        if len(colors) != 2:
            raise RuntimeError(
                "Color sorting rules require exactly two known tray colors."
            )
        if rule == "first-color":
            priority, other = colors
        elif rule == "second-color":
            other, priority = colors
        else:
            raise ValueError(f"Unknown color sorting constraint: {rule!r}.")
        return [
            f"Sort all {priority} cubes before sorting any {other} cube."
        ]

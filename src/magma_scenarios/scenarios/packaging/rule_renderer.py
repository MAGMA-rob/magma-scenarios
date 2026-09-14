from typing import List

from magma_core.simulation.state import RuleRenderer, TaskState
from magma_core.utils.text_utils import join_with_and

from .packaging_constraints import (
    DEFAULT_RECIPE_KEY,
    INCOMPATIBILITY_RULES_KEY,
)


class PackagingRuleRenderer(RuleRenderer):
    def rules(self, state: TaskState) -> List[str]:
        rules = []
        default_recipe = state.properties.get(DEFAULT_RECIPE_KEY, [])
        if default_recipe:
            rules.append(
                "The default tray recipe is "
                f"{join_with_and(list(default_recipe))}."
            )

        for first, second in state.properties.get(
            INCOMPATIBILITY_RULES_KEY,
            [],
        ):
            rules.append(
                f"{first} and {second} must not be placed on the same tray."
            )
        return rules

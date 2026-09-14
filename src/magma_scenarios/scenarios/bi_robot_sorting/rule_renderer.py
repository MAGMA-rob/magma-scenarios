from typing import List

from magma_core.simulation.state import RuleRenderer, TaskState

from .requests.common import TYPE_PRIORITY_KEY


class BiRobotSortingRuleRenderer(RuleRenderer):
    """Project the active fruit-type priority from scenario state."""

    def rules(self, state: TaskState) -> List[str]:
        priority_type = state.properties.get(TYPE_PRIORITY_KEY)
        if priority_type is None:
            return []
        return [
            f"Sort all {priority_type} objects before sorting other fruit types."
        ]

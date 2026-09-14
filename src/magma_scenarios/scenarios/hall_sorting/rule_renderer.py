from collections import defaultdict
from typing import List

from magma_core.simulation.state import RuleRenderer, TaskState
from magma_core.utils.text_utils import join_with_and

from .requests.common import PRIORITY_HALL_KEY, TYPE_HALL_RELATION


class HallSortingRuleRenderer(RuleRenderer):
    def rules(self, state: TaskState) -> List[str]:
        grouped = defaultdict(list)
        for object_type, hall in sorted(
            state.relations.get(TYPE_HALL_RELATION, {}).items()
        ):
            grouped[hall].append(object_type)
        rules = [
            f"{join_with_and(object_types)} objects belong in {hall}."
            for hall, object_types in sorted(grouped.items())
        ]
        priority_hall = state.properties.get(PRIORITY_HALL_KEY)
        if priority_hall is not None:
            rules.append(f"Complete {priority_hall} before other storage halls.")
        return rules

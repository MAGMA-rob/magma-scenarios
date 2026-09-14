from collections import defaultdict
from typing import Dict, Iterable, List

from magma_core.simulation.state import RuleRenderer, TaskState

from .laundry_constraints import CLOTHE_DETERGENT_KEY


def _join_values(values: Iterable[str]) -> str:
    values = list(values)
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


class LaundryRuleRenderer(RuleRenderer):
    """Project clothes-to-detergent relations into canonical rules."""

    def rules(self, state: TaskState) -> List[str]:
        grouped: Dict[str, List[str]] = defaultdict(list)
        for clothe, detergent in sorted(
            state.relations.get(CLOTHE_DETERGENT_KEY, {}).items()
        ):
            grouped[detergent].append(clothe)

        return sorted(
            f"{_join_values(sorted(clothes))} use {detergent}."
            for detergent, clothes in grouped.items()
        )

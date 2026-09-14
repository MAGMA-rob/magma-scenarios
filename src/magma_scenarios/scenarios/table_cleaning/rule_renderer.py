from typing import List

from magma_core.simulation.state import RuleRenderer, TaskState

from .common.planning import requirements_text


class TableCleaningRuleRenderer(RuleRenderer):
    """Project the editable table composition rule, when one exists."""

    def rules(self, state: TaskState) -> List[str]:
        requirements = state.relations.get("table_requirements", {})
        if not requirements:
            return []
        ordered = dict(sorted(requirements.items()))
        return [
            "When setting the table, put "
            f"{requirements_text(ordered)} on it."
        ]

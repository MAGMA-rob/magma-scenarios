from typing import Dict

from magma_core.simulation.state import TaskState

from magma_scenarios.templates.constraints import RelationAssignmentConstraint


CLOTHE_DETERGENT_KEY = "clothe_detergent"


def get_clothe_detergent_relations(state: TaskState) -> Dict[str, str]:
    if CLOTHE_DETERGENT_KEY not in state.relations:
        state.relations[CLOTHE_DETERGENT_KEY] = {}
    return state.relations[CLOTHE_DETERGENT_KEY]


class ClotheDetergentConstraint(RelationAssignmentConstraint):

    def __init__(self, clothe: str, detergent: str) -> None:
        super().__init__(
            source_value=clothe,
            target_value=detergent,
            relation_key=CLOTHE_DETERGENT_KEY,
            source_attribute_key="clothes",
            target_attribute_key="detergents",
        )


ClotheDetergentConstraints = ClotheDetergentConstraint

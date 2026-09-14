from typing import Dict, List, Sequence, Tuple

from magma_core.simulation.state import TaskState
from magma_core.utils.text_utils import join_with_and
from magma_scenarios.templates.requests import GiveRelationAssignmentRequest
from magma_scenarios.templates.requests.assignment_requests import (
    AssignmentSamplingMode,
)

from ..laundry_constraints import CLOTHE_DETERGENT_KEY, ClotheDetergentConstraint


class AssignClotheDetergentRequest(GiveRelationAssignmentRequest):
    """Sample permanent clothes-to-detergent compatibility rules."""

    def __init__(
        self,
        max_clothes_assignment: int = 9,
        assignment_modes: Sequence[AssignmentSamplingMode] = ("sample",),
    ) -> None:
        super().__init__(
            relation_key=CLOTHE_DETERGENT_KEY,
            source_attribute_key="clothes",
            target_attribute_key="detergents",
            max_simultaneous_change=max_clothes_assignment,
            assignment_modes=assignment_modes,
            intro_message="Hello, please remember that ",
            assignment_template="{source} uses {target}",
            constraint_builder=ClotheDetergentConstraint,
            constraint_message_builder=_build_grouped_assignment_message,
        )

    def sampling_weight(self, state: TaskState) -> float:
        if state.properties.get(
            f"{CLOTHE_DETERGENT_KEY}_needs_application",
            False,
        ):
            return 0.25
        return super().sampling_weight(state)


def _build_grouped_assignment_message(
    intro_message: str,
    assignments: List[Tuple[str, str]],
) -> str:
    grouped: Dict[str, List[str]] = {}
    for clothe, detergent in assignments:
        if detergent not in grouped:
            grouped[detergent] = []
        grouped[detergent].append(clothe)

    parts = [
        f"{join_with_and(clothes)} use {detergent}"
        for detergent, clothes in grouped.items()
    ]
    return intro_message + ", ".join(parts) + "."

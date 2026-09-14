import random
from typing import Dict, List

from magma_core.simulation.state import TaskState
from magma_scenarios.templates.requests.assignment_requests import (
    AssignmentPlan,
    AssignmentRecord,
    GiveRelationAssignmentRequest,
)


class GiveTypeHallAssignmentRequest(GiveRelationAssignmentRequest):
    """
    Create and update balanced type-to-hall rules.

    With four object types and two storage halls, exactly two types are
    always assigned to each hall.
    """

    reset_at_end = False

    def __init__(
        self,
        existing_relation_sampling_weight: float = 1.0,
        pending_relation_sampling_weight: float = 0.75,
    ) -> None:
        super().__init__(
            relation_key="type_hall",
            source_attribute_key="object_classes",
            target_attribute_key="storage_halls",

            # The custom sampling below manages all four types itself.
            max_simultaneous_change=4,

            empty_relation_sampling_weight=1.0,
            existing_relation_sampling_weight=(
                existing_relation_sampling_weight
            ),
            pending_relation_sampling_weight=(
                pending_relation_sampling_weight
            ),

            intro_message="Current storage rules: ",

            assignment_template=(
                "{source} objects belong in {target}"
            ),
            plural_assignment_template=(
                "{sources} objects belong in {target}"
            ),
            source_label_singular="object type",
            source_label_plural="object types",

            assignment_modes=("sample",),
        )

    @staticmethod
    def _is_balanced(sources: List[str], targets: List[str], current_assignments: Dict[str, str]) -> bool:
        if set(current_assignments) != set(sources):
            return False

        expected_count = len(sources) // len(targets)

        return all(
            sum(
                current_assignments.get(source) == target
                for source in sources
            )
            == expected_count
            for target in targets
        )

    @staticmethod
    def _build_initial_balanced_plan(sources: List[str], targets: List[str], current_assignments: Dict[str, str]) -> AssignmentPlan:
        shuffled_sources = sources.copy()
        random.shuffle(shuffled_sources)

        types_per_hall = len(shuffled_sources) // len(targets)

        records = []

        for index, source in enumerate(shuffled_sources):
            target_index = index // types_per_hall
            target = targets[target_index]

            if current_assignments.get(source) == target:
                continue

            records.append(
                AssignmentRecord(
                    source=source,
                    target=target,
                    previous_target=current_assignments.get(source),
                )
            )

        return AssignmentPlan(
            mode="sample",
            records=records,
            all_source_count=len(sources),
        )

    @staticmethod
    def _build_swap_plan(sources: List[str], targets: List[str], current_assignments: Dict[str, str]) -> AssignmentPlan:
        first_target, second_target = targets

        first_target_sources = [
            source
            for source in sources
            if current_assignments[source] == first_target
        ]

        second_target_sources = [
            source
            for source in sources
            if current_assignments[source] == second_target
        ]

        first_source = random.choice(first_target_sources)
        second_source = random.choice(second_target_sources)

        return AssignmentPlan(
            mode="sample",
            records=[
                AssignmentRecord(
                    source=first_source,
                    target=second_target,
                    previous_target=first_target,
                ),
                AssignmentRecord(
                    source=second_source,
                    target=first_target,
                    previous_target=second_target,
                ),
            ],
            all_source_count=len(sources),
        )

    def _sample_assignment_plan(self, state: TaskState) -> AssignmentPlan:
        sources = self._get_source_values(state)
        targets = self._get_target_values(state)

        if len(sources) != 4:
            raise RuntimeError(
                "GiveTypeHallAssignmentRequest requires exactly "
                f"4 object types, got {len(sources)}."
            )

        if len(targets) != 2:
            raise RuntimeError(
                "GiveTypeHallAssignmentRequest requires exactly "
                f"2 storage halls, got {len(targets)}."
            )

        current_assignments = state.relations.get(
            self.relation_key,
            {},
        )

        if self._is_balanced(
            sources=sources,
            targets=targets,
            current_assignments=current_assignments,
        ):
            return self._build_swap_plan(
                sources=sources,
                targets=targets,
                current_assignments=current_assignments,
            )

        return self._build_initial_balanced_plan(
            sources=sources,
            targets=targets,
            current_assignments=current_assignments,
        )

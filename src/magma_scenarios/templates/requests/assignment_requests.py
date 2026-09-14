from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple
import random
from collections import defaultdict

from magma_core.simulation.state.task_state import TaskState
from magma_scenarios.templates.requests.interact_request import (
    BaseConstraintRequest,
    ConstraintParameters,
)
from magma_core.simulation.constraints import BaseConstraint

from ..constraints import (
    RelationAssignmentConstraint,
    RelationDefaultConstraint,
)

ConstraintBuilder = Callable[[str, str], BaseConstraint]
AssignmentMessageBuilder = Callable[[str, List[Tuple[str, str]]], str]
ValueProvider = Callable[[TaskState], List[str]]
AssignmentSamplingMode = Literal["sample", "all_to_one", "split_one_vs_rest"]


@dataclass(frozen=True)
class AssignmentRecord:
    source: str
    target: str
    previous_target: Optional[str]


@dataclass(frozen=True)
class AssignmentPlan:
    mode: AssignmentSamplingMode
    records: List[AssignmentRecord]
    all_source_count: int


class GiveRelationAssignmentRequest(BaseConstraintRequest):
    """Sample source-to-target relation rules and mark them for later use.

    After the constraints are applied, ``{relation_key}_needs_application`` is
    set in state properties so follow-up requests can prioritize exercising the
    fresh or changed rule.
    """

    def __init__(
            self,
            relation_key: str,
            source_attribute_key: Optional[str],
            target_attribute_key: Optional[str],
            max_simultaneous_change: int = 1,
            empty_relation_sampling_weight: float = 3,
            existing_relation_sampling_weight: float = 1,
            pending_relation_sampling_weight: float = 0.25,
            intro_message: str = "Hey, here are some rules: ",
            assignment_template: str = "{source} goes to {target}",
            plural_assignment_template: Optional[str] = None,
            all_assignment_template: Optional[str] = None,
            rest_assignment_template: Optional[str] = None,
            source_label_singular: Optional[str] = None,
            source_label_plural: Optional[str] = None,
            assignment_modes: Sequence[AssignmentSamplingMode] = ("sample",),
            target_values: Optional[Sequence[str]] = None,
            source_values_provider: Optional[ValueProvider] = None,
            target_values_provider: Optional[ValueProvider] = None,
            constraint_builder: Optional[ConstraintBuilder] = None,
            constraint_message_builder: Optional[AssignmentMessageBuilder] = None,
        ) -> None:
        super().__init__()
        self.relation_key = relation_key
        self.source_attribute_key = source_attribute_key
        self.target_attribute_key = target_attribute_key
        self.max_change = max_simultaneous_change
        self.empty_relation_sampling_weight = empty_relation_sampling_weight
        self.existing_relation_sampling_weight = existing_relation_sampling_weight
        self.pending_relation_sampling_weight = pending_relation_sampling_weight
        self.intro_message = intro_message
        self.assignment_template = assignment_template
        self.plural_assignment_template = plural_assignment_template
        self.all_assignment_template = all_assignment_template
        self.rest_assignment_template = rest_assignment_template
        self.source_label_singular = source_label_singular or "source"
        self.source_label_plural = source_label_plural or source_attribute_key or "sources"
        self.assignment_modes = tuple(assignment_modes)
        self.target_values = list(target_values) if target_values is not None else None
        self.source_values_provider = source_values_provider
        self.target_values_provider = target_values_provider
        self.constraint_builder = constraint_builder
        self.constraint_message_builder = constraint_message_builder

        if len(self.assignment_modes) <= 0:
            raise ValueError("At least one assignment sampling mode must be provided")
        if self.existing_relation_sampling_weight < 0:
            raise ValueError("existing_relation_sampling_weight must be non-negative")
        if self.pending_relation_sampling_weight < 0:
            raise ValueError("pending_relation_sampling_weight must be non-negative")

    def _build_constraint(
        self,
        source_value: str,
        target_value: str,
    ) -> BaseConstraint:
        if self.constraint_builder is not None:
            return self.constraint_builder(source_value, target_value)
        return RelationAssignmentConstraint(
            source_value=source_value,
            target_value=target_value,
            relation_key=self.relation_key,
            source_attribute_key=self.source_attribute_key,
            target_attribute_key=self.target_attribute_key,
        )

    def _unique_values(self, values: Sequence[str]) -> List[str]:
        unique_values: List[str] = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
        return unique_values

    def _get_source_values(self, state: TaskState) -> List[str]:
        if self.source_values_provider is not None:
            return self._unique_values(self.source_values_provider(state))
        if self.source_attribute_key is None:
            return []
        return self._unique_values(state.attributes.get(self.source_attribute_key, []).copy())

    def _get_target_values(self, state: TaskState) -> List[str]:
        if self.target_values_provider is not None:
            return self._unique_values(self.target_values_provider(state))
        if self.target_values is not None:
            return self._unique_values(self.target_values)
        if self.target_attribute_key is None:
            return []
        return self._unique_values(state.attributes.get(self.target_attribute_key, []).copy())

    def _target_candidates_for_source(
            self,
            source_value: str,
            target_values: List[str],
            current_assignments: Dict[str, str],
        ) -> List[str]:
        current_target = current_assignments.get(source_value)
        candidates = [target for target in target_values if target != current_target]
        return candidates if candidates else target_values

    def _has_possible_assignment(self, state: TaskState) -> bool:
        source_values = self._get_source_values(state)
        target_values = self._get_target_values(state)
        if len(source_values) <= 0 or len(target_values) <= 0:
            return False

        current_assignments = state.relations.get(self.relation_key, {})
        return any(
            any(target != current_assignments.get(source) for target in target_values)
            for source in source_values
        )

    def sampling_weight(self, state: TaskState) -> float:
        if not self._has_possible_assignment(state):
            return 0
        if state.properties.get(f"{self.relation_key}_needs_application", False):
            return self.pending_relation_sampling_weight
        if len(state.relations.get(self.relation_key, {})) < 1:
            return self.empty_relation_sampling_weight
        return self.existing_relation_sampling_weight

    def apply_request(
        self,
        state: TaskState,
        parameters: ConstraintParameters,
    ) -> TaskState:
        state = super().apply_request(state, parameters)
        if parameters.constraints:
            state.properties[f"{self.relation_key}_needs_application"] = True
            pending_sources = [
                constraint.source_value
                for constraint in parameters.constraints
                if hasattr(constraint, "source_value")
            ]
            if pending_sources:
                state.properties[f"{self.relation_key}_pending_sources"] = pending_sources
        return state

    def _sample_individual_assignments(
            self,
            all_sources: List[str],
            all_targets: List[str],
            current_assignments: Dict[str, str],
        ) -> Optional[AssignmentPlan]:
        eligible_sources = [
            source
            for source in all_sources
            if any(target != current_assignments.get(source) for target in all_targets)
        ]
        if len(eligible_sources) <= 0:
            return None

        max_val = min(self.max_change, len(eligible_sources))
        nb_change = random.randint(1, max_val)
        random.shuffle(eligible_sources)

        records = []
        for source_value in eligible_sources[:nb_change]:
            target_value = random.choice(
                self._target_candidates_for_source(
                    source_value,
                    all_targets,
                    current_assignments,
                )
            )
            records.append(
                AssignmentRecord(
                    source=source_value,
                    target=target_value,
                    previous_target=current_assignments.get(source_value),
                )
            )

        return AssignmentPlan(
            mode="sample",
            records=records,
            all_source_count=len(all_sources),
        )

    def _sample_all_to_one_assignment(
            self,
            all_sources: List[str],
            all_targets: List[str],
            current_assignments: Dict[str, str],
        ) -> Optional[AssignmentPlan]:
        target_candidates = [
            target
            for target in all_targets
            if any(current_assignments.get(source) != target for source in all_sources)
        ]
        if len(target_candidates) <= 0:
            return None

        target_value = random.choice(target_candidates)
        return AssignmentPlan(
            mode="all_to_one",
            records=[
                AssignmentRecord(
                    source=source_value,
                    target=target_value,
                    previous_target=current_assignments.get(source_value),
                )
                for source_value in all_sources
            ],
            all_source_count=len(all_sources),
        )

    def _sample_split_assignment(
            self,
            all_sources: List[str],
            all_targets: List[str],
            current_assignments: Dict[str, str],
        ) -> Optional[AssignmentPlan]:
        if len(all_sources) < 2 or len(all_targets) < 2:
            return None

        shuffled_sources = all_sources.copy()
        shuffled_targets = all_targets.copy()
        random.shuffle(shuffled_sources)
        random.shuffle(shuffled_targets)

        for special_source in shuffled_sources:
            special_target_options = self._target_candidates_for_source(
                special_source,
                shuffled_targets,
                current_assignments,
            )
            random.shuffle(special_target_options)

            for special_target in special_target_options:
                rest_target_options = [
                    target
                    for target in shuffled_targets
                    if target != special_target
                ]
                random.shuffle(rest_target_options)

                for rest_target in rest_target_options:
                    records = [
                        AssignmentRecord(
                            source=special_source,
                            target=special_target,
                            previous_target=current_assignments.get(special_source),
                        )
                    ]
                    for source_value in all_sources:
                        if source_value == special_source:
                            continue
                        records.append(
                            AssignmentRecord(
                                source=source_value,
                                target=rest_target,
                                previous_target=current_assignments.get(source_value),
                            )
                        )

                    if any(record.previous_target != record.target for record in records):
                        return AssignmentPlan(
                            mode="split_one_vs_rest",
                            records=records,
                            all_source_count=len(all_sources),
                        )

        return None

    def _sample_assignment_plan(self, state: TaskState) -> AssignmentPlan:
        all_sources = self._get_source_values(state)
        all_targets = self._get_target_values(state)

        if len(all_sources) <= 0 or len(all_targets) <= 0:
            source_key = self.source_attribute_key or "source values"
            target_key = self.target_attribute_key or "target values"
            raise RuntimeError(
                f"Failed to build the stage from {self.__class__.__name__} "
                f"due to empty {source_key} or {target_key}"
            )

        current_assignments = state.relations.get(self.relation_key, {})
        modes_to_try = list(self.assignment_modes)
        preferred_mode = random.choice(modes_to_try)
        modes_to_try.remove(preferred_mode)
        random.shuffle(modes_to_try)
        modes_to_try.insert(0, preferred_mode)

        for mode in modes_to_try:
            if mode == "sample":
                plan = self._sample_individual_assignments(
                    all_sources,
                    all_targets,
                    current_assignments,
                )
            elif mode == "all_to_one":
                plan = self._sample_all_to_one_assignment(
                    all_sources,
                    all_targets,
                    current_assignments,
                )
            elif mode == "split_one_vs_rest":
                plan = self._sample_split_assignment(
                    all_sources,
                    all_targets,
                    current_assignments,
                )
            else:
                raise ValueError(f"Unsupported assignment mode: {mode}")

            if plan is not None and len(plan.records) > 0:
                return plan

        raise RuntimeError(
            f"Failed to build the stage from {self.__class__.__name__} "
            f"because no non-empty assignment could be sampled"
        )

    def _join_sources(self, sources: List[str]) -> str:
        if len(sources) == 1:
            return sources[0]
        if len(sources) == 2:
            return f"{sources[0]} and {sources[1]}"
        return ", ".join(sources[:-1]) + f", and {sources[-1]}"

    def _format_singular_assignment(self, source_value: str, target_value: str) -> str:
        return self.assignment_template.format(
            source=source_value,
            target=target_value,
        )

    def _format_plural_assignment(self, source_values: List[str], target_value: str) -> str:
        if self.plural_assignment_template is None:
            return ", ".join(
                self._format_singular_assignment(source, target_value)
                for source in source_values
            )
        return self.plural_assignment_template.format(
            sources=self._join_sources(source_values),
            target=target_value,
            source_label_singular=self.source_label_singular,
            source_label_plural=self.source_label_plural,
        )

    def _format_all_assignment(self, target_value: str) -> str:
        template = self.all_assignment_template or "all {source_label_plural} go to {target}"
        return template.format(
            target=target_value,
            source_label_singular=self.source_label_singular,
            source_label_plural=self.source_label_plural,
        )

    def _format_rest_assignment(self, target_value: str) -> str:
        template = self.rest_assignment_template or "every other {source_label_singular} goes to {target}"
        return template.format(
            target=target_value,
            source_label_singular=self.source_label_singular,
            source_label_plural=self.source_label_plural,
        )

    def _format_grouped_records(self, records: List[AssignmentRecord], all_source_count: int) -> List[str]:
        grouped_assignments: Dict[str, List[str]] = defaultdict(list)
        for record in records:
            grouped_assignments[record.target].append(record.source)

        clauses = []
        for target_value, source_values in grouped_assignments.items():
            if len(source_values) == all_source_count and all_source_count > 1:
                clauses.append(self._format_all_assignment(target_value))
            elif len(source_values) == 1:
                clauses.append(
                    self._format_singular_assignment(source_values[0], target_value)
                )
            else:
                clauses.append(self._format_plural_assignment(source_values, target_value))
        return clauses

    def _choose_message_prefix(self, plan: AssignmentPlan) -> str:
        has_override = any(
            record.previous_target is not None
            and record.previous_target != record.target
            for record in plan.records
        )
        has_new_assignment = any(record.previous_target is None for record in plan.records)

        if plan.mode in ("all_to_one", "split_one_vs_rest"):
            return random.choice((
                "From now on, ",
                "New rule: ",
                "Please update the rules: ",
            ))
        if has_override and has_new_assignment:
            return random.choice((
                "Please update the rules: ",
                "Use these rules from now on: ",
                self.intro_message,
            ))
        if has_override:
            return random.choice((
                "From now on, ",
                "Small update to the current rules: ",
                "Modification of the existing rule: ",
            ))
        return random.choice((
            self.intro_message,
            "New rule: ",
            "Please remember that ",
        ))

    def _build_default_message(self, plan: AssignmentPlan) -> str:
        if plan.mode == "all_to_one":
            clause = self._format_all_assignment(plan.records[0].target)
        elif plan.mode == "split_one_vs_rest":
            special_record = plan.records[0]
            rest_target = plan.records[1].target
            clause = (
                self._format_singular_assignment(
                    special_record.source,
                    special_record.target,
                )
                + ", and "
                + self._format_rest_assignment(rest_target)
            )
        else:
            clause = ", ".join(
                self._format_grouped_records(plan.records, plan.all_source_count)
            )

        return self._choose_message_prefix(plan) + clause + "."

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        constraints: List[BaseConstraint] = []
        plan = self._sample_assignment_plan(state)
        assignments = [(record.source, record.target) for record in plan.records]
        default_target = None
        if self.constraint_message_builder is None:
            if plan.mode == "all_to_one":
                default_target = plan.records[0].target
            elif plan.mode == "split_one_vs_rest":
                default_target = plan.records[1].target
            elif (
                len(plan.records) == plan.all_source_count
                and len({record.target for record in plan.records}) == 1
            ):
                default_target = plan.records[0].target

        if default_target is not None:
            constraints.append(
                RelationDefaultConstraint(
                    relation_key=self.relation_key,
                    target_value=default_target,
                    target_attribute_key=self.target_attribute_key,
                )
            )

        for source_value, target_value in assignments:
            constraints.append(
                self._build_constraint(
                    source_value,
                    target_value,
                )
            )

        if self.constraint_message_builder is not None:
            instruction = self.constraint_message_builder(
                self.intro_message,
                assignments,
            )
        else:
            instruction = self._build_default_message(plan)
        return ConstraintParameters(constraints, instruction)


class GiveObjectAssignmentRequest(GiveRelationAssignmentRequest):
    """Sample direct object-to-area rules and expose them as one constraint request."""

    def __init__(
            self,
            max_simultaneous_change : int = 1,
            existing_relation_sampling_weight: float = 1,
            pending_relation_sampling_weight: float = 0.25,
            assignment_modes: Sequence[AssignmentSamplingMode] = (
                "sample",
                "sample",
                "all_to_one",
                "split_one_vs_rest",
            ),
        ):
        super().__init__(
            relation_key="object_area",
            source_attribute_key="objects",
            target_attribute_key="target_areas",
            max_simultaneous_change=max_simultaneous_change,
            existing_relation_sampling_weight=existing_relation_sampling_weight,
            pending_relation_sampling_weight=pending_relation_sampling_weight,
            intro_message="Hey, here are some sorting rules: ",
            assignment_template="{source} goes to {target}",
            plural_assignment_template="{sources} go to {target}",
            all_assignment_template="all {source_label_plural} go to {target}",
            rest_assignment_template="every other {source_label_singular} goes to {target}",
            source_label_singular="object",
            source_label_plural="objects",
            assignment_modes=assignment_modes,
        )

class GiveObjectCategoryRequest(GiveRelationAssignmentRequest):
    """Sample object-to-category updates for sorting tasks."""

    categories : List[str]

    def __init__(
            self,
            available_categories : List[str],
            max_object_assignment : int = 2,
            existing_relation_sampling_weight: float = 1,
            pending_relation_sampling_weight: float = 0.25,
            assignment_modes: Sequence[AssignmentSamplingMode] = (
                "sample",
                "sample",
                "all_to_one",
                "split_one_vs_rest",
            ),
        ):
        if len(available_categories) < 2:
            raise ValueError("It must have at least 2 categories")
        self.categories = available_categories
        self.max_obj = max_object_assignment
        super().__init__(
            relation_key="object_type",
            source_attribute_key="objects",
            target_attribute_key=None,
            target_values=available_categories,
            max_simultaneous_change=max_object_assignment,
            empty_relation_sampling_weight=4,
            existing_relation_sampling_weight=existing_relation_sampling_weight,
            pending_relation_sampling_weight=pending_relation_sampling_weight,
            intro_message="Hello, ",
            assignment_template="{source} is {target}",
            plural_assignment_template="{sources} are {target}",
            all_assignment_template="all {source_label_plural} are {target}",
            rest_assignment_template="every other {source_label_singular} is {target}",
            source_label_singular="object",
            source_label_plural="objects",
            assignment_modes=assignment_modes,
        )


def _get_known_object_categories(state: TaskState) -> List[str]:
    known_objects = set(state.attributes.get("objects", []))
    categories: List[str] = []
    for object_name, category in state.relations.get("object_type", {}).items():
        if object_name not in known_objects:
            continue
        if category not in categories:
            categories.append(category)
    return categories


class GiveCategoryAssignmentRequest(GiveRelationAssignmentRequest):
    """Sample category-to-area routing rules for sorting tasks."""

    categories : List[str]

    def __init__(
            self,
            available_categories : List[str],
            max_categories_assignment : int = 2,
            existing_relation_sampling_weight: float = 1,
            pending_relation_sampling_weight: float = 0.25,
            assignment_modes: Sequence[AssignmentSamplingMode] = (
                "sample",
                "sample",
                "all_to_one",
                "split_one_vs_rest",
            ),
        ):
        if len(available_categories) < 2:
            raise ValueError("It must have at least 2 categories")
        self.categories = available_categories
        self.max_categories = max_categories_assignment
        super().__init__(
            relation_key="type_area",
            source_attribute_key=None,
            target_attribute_key="target_areas",
            source_values_provider=_get_known_object_categories,
            max_simultaneous_change=max_categories_assignment,
            empty_relation_sampling_weight=4,
            existing_relation_sampling_weight=existing_relation_sampling_weight,
            pending_relation_sampling_weight=pending_relation_sampling_weight,
            intro_message="Hello, ",
            assignment_template="{source} goes to {target}",
            plural_assignment_template="{sources} go to {target}",
            all_assignment_template="all {source_label_plural} go to {target}",
            rest_assignment_template="every other {source_label_singular} goes to {target}",
            source_label_singular="category",
            source_label_plural="categories",
            assignment_modes=assignment_modes,
        )

    def sampling_weight(self, state: TaskState) -> float:
        if len(state.relations.get("object_type",{})) < 1:
            return 0
        if len(state.relations.get("type_area",{})) > 3:
            return 0 # AVoiding too much category
        return super().sampling_weight(state)

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
import random
from typing import Dict, Iterable, List, Tuple

from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest

from ...common.attributes import (
    ADVANCED_LOCATIONS,
    CLEAN_STATE,
    DIRTY_STATE,
    dishware,
    food,
)
from ...common.planning import object_type, requirements_text
from ..adct_stages import ArrangeAndCleanObjectsStage
from ..environment_transitions import ResetFoodTransition
from .planning import ParallelPlan, assign_target_objects, build_parallel_plan


AssignmentCounts = Dict[str, Dict[str, Dict[int, int]]]
FrozenAssignment = Tuple[
    Tuple[str, Tuple[Tuple[str, Tuple[Tuple[int, int], ...]], ...]], ...
]


@dataclass(frozen=True)
class AdvancedExecutionPlan:
    active_objects: Tuple[str, ...]
    target_assignment: FrozenAssignment
    target_locations: Tuple[Tuple[str, str], ...]
    target_states: Tuple[Tuple[str, int], ...]
    initial_correct: int
    plan: ParallelPlan


@dataclass(frozen=True)
class SetTableParameters:
    execution: AdvancedExecutionPlan
    requirements: Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class CleanAndStoreParameters:
    execution: AdvancedExecutionPlan


def _empty_assignment() -> AssignmentCounts:
    return {location: {} for location in ADVANCED_LOCATIONS}


def _add_assignment(
    assignment: AssignmentCounts,
    location: str,
    object_type: str,
    state: int,
    count: int = 1,
) -> None:
    state_counts = assignment[location].setdefault(object_type, {})
    state_counts[state] = state_counts.get(state, 0) + count


def _assignment_total(assignment: AssignmentCounts) -> int:
    return sum(
        count
        for type_counts in assignment.values()
        for state_counts in type_counts.values()
        for count in state_counts.values()
    )


def _matching_count(current: AssignmentCounts, target: AssignmentCounts) -> int:
    return sum(
        min(
            current.get(location, {})
            .get(object_type, {})
            .get(state, 0),
            target_count,
        )
        for location, type_counts in target.items()
        for object_type, state_counts in type_counts.items()
        for state, target_count in state_counts.items()
    )


def _freeze_assignment(assignment: AssignmentCounts) -> FrozenAssignment:
    return tuple(
        (
            location,
            tuple(
                (
                    object_type,
                    tuple(state_counts.items()),
                )
                for object_type, state_counts in type_counts.items()
            ),
        )
        for location, type_counts in assignment.items()
    )


def _thaw_assignment(assignment: FrozenAssignment) -> AssignmentCounts:
    return {
        location: {
            object_type: dict(state_counts)
            for object_type, state_counts in type_counts
        }
        for location, type_counts in assignment
    }


def _normalized_after_food_reset(
    assignment: AssignmentCounts,
) -> AssignmentCounts:
    normalized = _empty_assignment()
    food_types = {object_type(object_name) for object_name in food}
    for location, type_counts in assignment.items():
        for type_name, state_counts in type_counts.items():
            for state, count in state_counts.items():
                if type_name in food_types:
                    target = "food_storage" if location == "trashcan" else location
                    _add_assignment(
                        normalized,
                        target,
                        type_name,
                        CLEAN_STATE,
                        count,
                    )
                else:
                    _add_assignment(
                        normalized,
                        location,
                        type_name,
                        state,
                        count,
                    )
    return normalized


def _build_progressive_stages(
    target: AssignmentCounts,
    initial_correct: int,
    instruction: str,
    entry_transition=None,
) -> List[BaseTaskStage]:
    total_targets = _assignment_total(target)
    return [
        ArrangeAndCleanObjectsStage(
            minimum=minimum,
            required_state_counts=target,
            instruction=(
                instruction if minimum == initial_correct + 1 else "none"
            ),
            flag_answer=minimum == total_targets,
            entry_transition=(
                entry_transition if minimum == initial_correct + 1 else None
            ),
        )
        for minimum in range(initial_correct + 1, total_targets + 1)
    ]


def _available_slots(active_objects: Iterable[str]) -> List[str]:
    eligible = set(food + dishware)
    return [
        object_type(object_name)
        for object_name in active_objects
        if object_name in eligible
    ]


class SetTableRequest(BaseRequest[SetTableParameters]):
    def __init__(self, max_table_objects: int = 4, weight: float = 3.0) -> None:
        super().__init__()
        if max_table_objects < 2:
            raise ValueError("max_table_objects must be at least 2.")
        if weight < 0:
            raise ValueError("weight must be non-negative.")
        self.max_table_objects = max_table_objects
        self.weight = weight

    def _possible_requirements(self, state: TaskState) -> List[Dict[str, int]]:
        slots = _available_slots(state.properties.get("active_objects", []))
        maximum = min(self.max_table_objects, len(slots))
        if maximum < 2:
            return []
        current = state.relations.get("table_assignment", {})
        possibilities = {
            tuple(sorted(Counter(slots[index] for index in indices).items()))
            for count in range(2, maximum + 1)
            for indices in combinations(range(len(slots)), count)
        }
        current_key = tuple(sorted(current.items()))
        return [
            dict(requirements)
            for requirements in sorted(possibilities)
            if requirements != current_key
        ]

    def sampling_weight(self, state: TaskState) -> float:
        return self.weight if self._possible_requirements(state) else 0.0

    def sample_parameters(self, state: TaskState) -> SetTableParameters:
        possibilities = self._possible_requirements(state)
        if not possibilities:
            raise RuntimeError("No new feasible table setting is available.")
        requirements = random.choice(possibilities)
        active_objects = state.properties.get("active_objects", []).copy()
        available_by_type = Counter(_available_slots(active_objects))
        target = _empty_assignment()
        for object_type, available_count in available_by_type.items():
            table_count = requirements.get(object_type, 0)
            storage_target = (
                "food_storage"
                if any(
                    object_name.startswith(object_type + "_")
                    and object_name in food
                    for object_name in active_objects
                )
                else "dish_storage"
            )
            if table_count:
                _add_assignment(
                    target,
                    "table",
                    object_type,
                    CLEAN_STATE,
                    table_count,
                )
            if available_count > table_count:
                _add_assignment(
                    target,
                    storage_target,
                    object_type,
                    CLEAN_STATE,
                    available_count - table_count,
                )

        current = deepcopy(
            state.relations.get("object_assignment", _empty_assignment())
        )
        normalized_current = _normalized_after_food_reset(current)
        initial_correct = _matching_count(normalized_current, target)
        current_locations = state.relations.get("object_area", {}).copy()
        current_states = state.relations.get("object_state", {}).copy()
        for object_name in active_objects:
            if object_name in food:
                if current_locations.get(object_name) == "trashcan":
                    current_locations[object_name] = "food_storage"
                current_states[object_name] = CLEAN_STATE
        target_locations, target_states = assign_target_objects(
            active_objects,
            current_locations,
            current_states,
            target,
        )
        stage_count = _assignment_total(target) - initial_correct
        if stage_count <= 0:
            raise RuntimeError("The sampled table setting is already satisfied.")
        plan = build_parallel_plan(
            stage_count=stage_count,
            current_locations=current_locations,
            current_states=current_states,
            target_locations=target_locations,
            target_states=target_states,
        )
        return SetTableParameters(
            execution=AdvancedExecutionPlan(
                active_objects=tuple(active_objects),
                target_assignment=_freeze_assignment(target),
                target_locations=tuple(target_locations.items()),
                target_states=tuple(target_states.items()),
                initial_correct=initial_correct,
                plan=plan,
            ),
            requirements=tuple(requirements.items()),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: SetTableParameters,
    ) -> List[BaseTaskStage]:
        execution = parameters.execution
        instruction = (
            "Set the table with "
            f"{requirements_text(dict(parameters.requirements))}."
        )
        stages = _build_progressive_stages(
            _thaw_assignment(execution.target_assignment),
            execution.initial_correct,
            instruction,
            ResetFoodTransition(list(execution.active_objects)),
        )
        for stage, decision_count in zip(
            stages, execution.plan.decisions_per_stage
        ):
            stage.target_tool_calls = decision_count
            stage.max_tool_calls = max(
                stage.get_max_tool_calls() or 0,
                decision_count,
            )
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: SetTableParameters,
    ) -> TaskState:
        execution = parameters.execution
        requirements = dict(parameters.requirements)
        state.relations["object_assignment"] = _thaw_assignment(
            execution.target_assignment
        )
        state.relations["table_assignment"] = requirements
        state.relations["object_area"] = dict(execution.target_locations)
        state.properties["table_item_count"] = sum(requirements.values())
        state.properties["table_is_empty"] = False
        state.relations["object_state"] = dict(execution.target_states)
        return state


class CleanAndStoreRequest(BaseRequest[CleanAndStoreParameters]):
    def __init__(self, weight: float = 2.0) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError("weight must be non-negative.")
        self.weight = weight

    def _target(self, state: TaskState) -> AssignmentCounts:
        target = _empty_assignment()
        active_objects = state.properties.get("active_objects", [])
        object_states = state.relations.get("object_state", {})
        for object_name in active_objects:
            type_name = object_type(object_name)
            if object_name in food:
                state_value = object_states.get(object_name, CLEAN_STATE)
                location = (
                    "trashcan" if state_value == DIRTY_STATE else "food_storage"
                )
                _add_assignment(target, location, type_name, state_value)
            elif object_name in dishware:
                _add_assignment(
                    target,
                    "dish_storage",
                    type_name,
                    CLEAN_STATE,
                )
        return target

    def sampling_weight(self, state: TaskState) -> float:
        target = self._target(state)
        current = state.relations.get("object_assignment", _empty_assignment())
        return (
            self.weight
            if _matching_count(current, target) < _assignment_total(target)
            else 0.0
        )

    def sample_parameters(self, state: TaskState) -> CleanAndStoreParameters:
        target = self._target(state)
        current = state.relations.get("object_assignment", _empty_assignment())
        initial_correct = _matching_count(current, target)
        active_objects = state.properties.get("active_objects", []).copy()
        current_locations = state.relations.get("object_area", {}).copy()
        current_states = state.relations.get("object_state", {}).copy()
        target_locations, target_states = assign_target_objects(
            active_objects,
            current_locations,
            current_states,
            target,
        )
        stage_count = _assignment_total(target) - initial_correct
        if stage_count <= 0:
            raise RuntimeError("All objects are already clean and stored.")
        plan = build_parallel_plan(
            stage_count=stage_count,
            current_locations=current_locations,
            current_states=current_states,
            target_locations=target_locations,
            target_states=target_states,
        )
        return CleanAndStoreParameters(AdvancedExecutionPlan(
            active_objects=tuple(active_objects),
            target_assignment=_freeze_assignment(target),
            target_locations=tuple(target_locations.items()),
            target_states=tuple(target_states.items()),
            initial_correct=initial_correct,
            plan=plan,
        ))

    def create_stages(
        self,
        state: TaskState,
        parameters: CleanAndStoreParameters,
    ) -> List[BaseTaskStage]:
        execution = parameters.execution
        stages = _build_progressive_stages(
            _thaw_assignment(execution.target_assignment),
            execution.initial_correct,
            "Clean and store every object correctly.",
        )
        for stage, decision_count in zip(
            stages, execution.plan.decisions_per_stage
        ):
            stage.target_tool_calls = decision_count
            stage.max_tool_calls = max(
                stage.get_max_tool_calls() or 0,
                decision_count,
            )
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: CleanAndStoreParameters,
    ) -> TaskState:
        execution = parameters.execution
        state.relations["object_assignment"] = _thaw_assignment(
            execution.target_assignment
        )
        state.relations["table_assignment"] = {}
        state.relations["object_area"] = dict(execution.target_locations)
        state.properties["table_item_count"] = 0
        state.properties["table_is_empty"] = True
        state.relations["object_state"] = dict(execution.target_states)
        return state

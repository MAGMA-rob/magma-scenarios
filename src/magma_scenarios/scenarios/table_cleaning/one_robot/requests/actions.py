from collections import Counter
from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple

from magma_core.simulation.stage import BaseStageEnvironmentTransition, BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest

from ...common.attributes import CLEAN_STATE, DIRTY_STATE, dishware, food
from ...common.planning import object_type, requirements_text
from ..clean_stage import ClearTableStage, SetTableStage, WipeStage
from ..environment_transitions import (
    DirtyTableObjectsTransition,
    RestockStorageTransition,
)
from .common import available_requirement_slots


@dataclass(frozen=True)
class CleanTableParameters:
    active_objects: Tuple[str, ...]
    table_objects: Tuple[str, ...]
    dirty_objects: Tuple[str, ...]
    final_locations: Tuple[Tuple[str, str], ...]
    final_states: Tuple[Tuple[str, int], ...]
    includes_wiping: bool
    combined_instruction: bool

    @property
    def instruction(self) -> str:
        return (
            "Clean and wipe the table."
            if self.combined_instruction
            else "Clean the table."
        )


@dataclass(frozen=True)
class SetTablePlan:
    active_objects: Tuple[str, ...]
    requirements: Tuple[Tuple[str, int], ...]
    placed_objects: Tuple[str, ...]
    initial_locations: Tuple[Tuple[str, str], ...]
    first_minimum: int
    restock_before_start: bool


@dataclass(frozen=True)
class SetTableParameters(SetTablePlan):
    pass


@dataclass(frozen=True)
class DirectSetTableParameters(SetTablePlan):
    additions: Tuple[Tuple[str, int], ...]


def _set_instruction(parameters: SetTablePlan) -> str:
    if isinstance(parameters, SetTableParameters):
        return "Set the table."
    additions = dict(parameters.additions)
    verb = "Set the table with" if parameters.first_minimum == 1 else (
        "Complete the table by adding"
    )
    return f"{verb} {requirements_text(additions)}."


def _build_set_table_stages(
    active_objects: List[str],
    requirements: Dict[str, int],
    first_minimum: int,
    instruction: str,
    entry_transition: Optional[BaseStageEnvironmentTransition],
) -> List[BaseTaskStage]:
    total_required = sum(requirements.values())
    return [
        SetTableStage(
            active_objects=active_objects,
            requirements=requirements,
            minimum=minimum,
            instruction=instruction if minimum == first_minimum else "none",
            flag_answer=minimum == total_required,
            entry_transition=(
                entry_transition if minimum == first_minimum else None
            ),
        )
        for minimum in range(first_minimum, total_required + 1)
    ]


class CleanTableRequest(BaseRequest[CleanTableParameters]):
    def __init__(self, dirty_probability: float = 0.5) -> None:
        super().__init__()
        if not 0 <= dirty_probability <= 1:
            raise ValueError("dirty_probability must be between 0 and 1.")
        self.dirty_probability = dirty_probability

    def sampling_weight(self, state: TaskState) -> float:
        return 2.0 if state.properties.get("table_item_count", 0) > 0 else 0.0

    def sample_parameters(self, state: TaskState) -> CleanTableParameters:
        table_item_count = state.properties.get("table_item_count", 0)
        active_objects = state.properties.get("active_objects", []).copy()
        if table_item_count <= 0:
            raise RuntimeError("CleanTableRequest requires objects on the table.")

        cleaning_mode = random.randrange(3)
        includes_wiping = cleaning_mode != 2
        combined_instruction = cleaning_mode == 0
        object_locations = state.relations.get("object_area", {})
        object_states = state.relations.get("object_state", {})
        table_objects = sorted(
            object_name
            for object_name in active_objects
            if object_locations.get(object_name) == "table"
        )
        if len(table_objects) != table_item_count:
            raise RuntimeError(
                "CleanTableRequest requires exact object locations for planning."
            )

        dirty_objects = [
            object_name
            for object_name in table_objects
            if object_states.get(object_name, CLEAN_STATE) == DIRTY_STATE
            or random.random() < self.dirty_probability
        ]
        final_states = {
            object_name: (
                DIRTY_STATE
                if object_name in dirty_objects
                else object_states.get(object_name, CLEAN_STATE)
            )
            for object_name in table_objects
        }
        final_locations: Dict[str, str] = {}
        for object_name in table_objects:
            object_state = final_states[object_name]
            if object_name in food:
                final_locations[object_name] = (
                    "trashcan" if object_state == DIRTY_STATE else "food_storage"
                )
            elif object_name in dishware:
                final_locations[object_name] = (
                    "washing_machine"
                    if object_state == DIRTY_STATE
                    else "dish_storage"
                )
            else:
                raise RuntimeError(f"Unsupported table object {object_name}.")

        return CleanTableParameters(
            tuple(active_objects),
            tuple(table_objects),
            tuple(dirty_objects),
            tuple(final_locations.items()),
            tuple(final_states.items()),
            includes_wiping,
            combined_instruction,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: CleanTableParameters,
    ) -> List[BaseTaskStage]:
        table_item_count = len(parameters.table_objects)
        active_objects = list(parameters.active_objects)
        stages: List[BaseTaskStage] = []
        for index in range(table_item_count):
            moved_count = index + 1
            stages.append(
                ClearTableStage(
                    active_objects=active_objects,
                    maximum_remaining=table_item_count - moved_count,
                    instruction=(
                        parameters.instruction if moved_count == 1 else "none"
                    ),
                    flag_answer=(
                        moved_count == table_item_count
                        and not parameters.includes_wiping
                    ),
                    entry_transition=(
                        DirtyTableObjectsTransition(
                            dirty_objects=list(parameters.dirty_objects)
                        )
                        if moved_count == 1
                        else None
                    ),
                )
            )
        if parameters.includes_wiping:
            stages.append(
                WipeStage(
                    instruction=(
                        "none"
                        if parameters.combined_instruction
                        else "Now wipe the table."
                    ),
                    flag_answer=True,
                    reset_at_end=False,
                )
            )
        final_stage = stages[-1]
        final_stage.target_tool_calls = final_stage.get_target_tool_calls() + 1
        final_stage.max_tool_calls = max(
            final_stage.get_max_tool_calls() or 0,
            final_stage.target_tool_calls,
        )
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: CleanTableParameters,
    ) -> TaskState:
        state.properties["table_item_count"] = 0
        state.properties["table_is_empty"] = True
        state.relations["table_assignment"] = {}
        state.relations.setdefault("object_area", {}).update(
            dict(parameters.final_locations)
        )
        state.relations.setdefault("object_state", {}).update(
            dict(parameters.final_states)
        )
        return state


class SetTableRequest(BaseRequest[SetTableParameters]):
    def __init__(
        self,
        normal_weight: float = 3.0,
        pending_rule_weight: float = 6.0,
    ) -> None:
        super().__init__()
        if normal_weight < 0 or pending_rule_weight < 0:
            raise ValueError("Sampling weights must be non-negative.")
        self.normal_weight = normal_weight
        self.pending_rule_weight = pending_rule_weight

    def sampling_weight(self, state: TaskState) -> float:
        requirements = state.relations.get("table_requirements", {})
        if not state.properties.get("table_is_empty", False) or not requirements:
            return 0
        if state.properties.get("table_requirements_needs_application", False):
            return self.pending_rule_weight
        return self.normal_weight

    def sample_parameters(self, state: TaskState) -> SetTableParameters:
        if not state.properties.get("table_is_empty", False):
            raise RuntimeError("SetTableRequest requires an empty table.")
        requirements = state.relations.get("table_requirements", {}).copy()
        if not requirements:
            raise RuntimeError("SetTableRequest requires table requirements.")
        available = available_requirement_slots(
            state.properties.get("active_objects", [])
        )
        for requirement, count in requirements.items():
            if available.count(requirement) < count:
                raise RuntimeError(
                    f"The table requirement {requirement}={count} is not feasible."
                )
        active_objects = state.properties.get("active_objects", []).copy()
        remaining_by_type = Counter(requirements)
        placed_objects = []
        for object_name in sorted(active_objects):
            object_type = object_name.rsplit("_", 1)[0]
            if remaining_by_type[object_type] > 0:
                placed_objects.append(object_name)
                remaining_by_type[object_type] -= 1
        if any(remaining_by_type.values()):
            raise RuntimeError("Unable to select concrete objects for table setting.")
        initial_locations = {
            object_name: (
                "food_storage" if object_name in food else "dish_storage"
            )
            for object_name in active_objects
        }
        return SetTableParameters(
            active_objects=tuple(active_objects),
            requirements=tuple(requirements.items()),
            placed_objects=tuple(placed_objects),
            initial_locations=tuple(initial_locations.items()),
            first_minimum=1,
            restock_before_start=True,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: SetTablePlan,
    ) -> List[BaseTaskStage]:
        stages = _build_set_table_stages(
            list(parameters.active_objects),
            dict(parameters.requirements),
            parameters.first_minimum,
            _set_instruction(parameters),
            (
                RestockStorageTransition(list(parameters.active_objects))
                if parameters.restock_before_start
                else None
            ),
        )
        stages[-1].target_tool_calls += 1
        stages[-1].max_tool_calls = max(
            stages[-1].max_tool_calls,
            stages[-1].target_tool_calls,
        )
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: SetTableParameters,
    ) -> TaskState:
        requirements = dict(parameters.requirements)
        state.properties["table_item_count"] = sum(requirements.values())
        state.properties["table_is_empty"] = False
        state.relations["table_assignment"] = requirements
        if isinstance(parameters, SetTableParameters):
            state.properties["table_requirements_needs_application"] = False
            state.properties["table_requirements_applications"] = (
                state.properties.get("table_requirements_applications", 0) + 1
            )
        locations = state.relations.setdefault("object_area", {})
        states = state.relations.setdefault("object_state", {})
        placed_objects = set(parameters.placed_objects)
        for object_name in parameters.active_objects:
            locations[object_name] = (
                "table"
                if object_name in placed_objects
                else "food_storage"
                if object_name in food
                else "dish_storage"
            )
            states[object_name] = CLEAN_STATE
        return state


class DirectSetTableRequest(SetTableRequest):
    def __init__(
        self,
        max_table_objects: int = 4,
        empty_table_weight: float = 3.0,
        completion_weight: float = 2.0,
    ) -> None:
        super().__init__(0, 0)
        if max_table_objects < 1:
            raise ValueError("max_table_objects must be positive.")
        if min(empty_table_weight, completion_weight) < 0:
            raise ValueError("Sampling weights must be non-negative.")
        self.max_table_objects = max_table_objects
        self.empty_table_weight = empty_table_weight
        self.completion_weight = completion_weight

    def sampling_weight(self, state: TaskState) -> float:
        slots = available_requirement_slots(
            state.properties.get("active_objects", [])
        )
        table_item_count = state.properties.get("table_item_count", 0)
        maximum_table_size = min(self.max_table_objects, len(slots))
        if not slots or table_item_count >= maximum_table_size:
            return 0
        if state.properties.get("table_is_empty", False):
            return self.empty_table_weight
        return self.completion_weight

    def sample_parameters(self, state: TaskState) -> DirectSetTableParameters:
        slots = available_requirement_slots(
            state.properties.get("active_objects", [])
        )
        table_contents = Counter(state.relations.get("table_assignment", {}))
        table_item_count = state.properties.get("table_item_count", 0)
        if sum(table_contents.values()) != table_item_count:
            raise RuntimeError("The tracked table composition is inconsistent.")
        remaining_slots = list((Counter(slots) - table_contents).elements())
        available_capacity = min(
            self.max_table_objects - table_item_count,
            len(remaining_slots),
        )
        if available_capacity <= 0:
            raise RuntimeError("No object can be added to the table.")
        minimum_additions = (
            2 if table_item_count == 0 and available_capacity >= 2 else 1
        )
        addition_count = random.randint(minimum_additions, available_capacity)
        additions = Counter(random.sample(remaining_slots, k=addition_count))
        final_requirements = table_contents + additions

        active_objects = state.properties.get("active_objects", []).copy()
        remaining_additions = additions.copy()
        object_locations = state.relations.get("object_area", {})
        added_objects = []
        for object_name in sorted(active_objects):
            object_type = object_name.rsplit("_", 1)[0]
            if (
                object_locations.get(object_name) != "table"
                and remaining_additions[object_type] > 0
            ):
                added_objects.append(object_name)
                remaining_additions[object_type] -= 1
        if any(remaining_additions.values()):
            raise RuntimeError("Unable to select concrete additions for the table.")

        if table_item_count == 0:
            initial_locations = {
                object_name: (
                    "food_storage" if object_name in food else "dish_storage"
                )
                for object_name in active_objects
            }
        else:
            initial_locations = object_locations.copy()
        return DirectSetTableParameters(
            active_objects=tuple(active_objects),
            requirements=tuple(final_requirements.items()),
            placed_objects=tuple(added_objects),
            initial_locations=tuple(initial_locations.items()),
            first_minimum=table_item_count + 1,
            restock_before_start=table_item_count == 0,
            additions=tuple(additions.items()),
        )

    def apply_request(
        self,
        state: TaskState,
        parameters: DirectSetTableParameters,
    ) -> TaskState:
        requirements = dict(parameters.requirements)
        state.relations["table_assignment"] = requirements
        state.properties["table_item_count"] = sum(requirements.values())
        state.properties["table_is_empty"] = False
        locations = state.relations.setdefault("object_area", {})
        states = state.relations.setdefault("object_state", {})
        for object_name in parameters.placed_objects:
            locations[object_name] = "table"
            states[object_name] = CLEAN_STATE
        return state

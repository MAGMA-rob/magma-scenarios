from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from magma_core.domain import Call

from ...common.attributes import (
    ADVANCED_LOCATIONS,
    CLEAN_STATE,
    DIRTY_STATE,
    cleaning_objects,
    dishware,
    food,
)
from ..adct_tools import DISH_ROBOT, TABLE_ROBOT


AssignmentCounts = Dict[str, Dict[str, Dict[int, int]]]


@dataclass(frozen=True)
class PlannedOperation:
    robot: str
    call: Call
    feedback: str


@dataclass(frozen=True)
class ScheduledBatch:
    operations: Tuple[PlannedOperation, ...]
    objects: Tuple[str, ...]
    completed_objects: Tuple[str, ...]


@dataclass(frozen=True)
class ParallelPlan:
    batches: Tuple[ScheduledBatch, ...]
    decisions_per_stage: Tuple[int, ...]

    @property
    def planned_objects(self) -> Tuple[str, ...]:
        return tuple(sorted({
            object_name
            for batch in self.batches
            for object_name in batch.objects
        }))


def assign_target_objects(
    active_objects: Sequence[str],
    current_locations: Dict[str, str],
    current_states: Dict[str, int],
    target: AssignmentCounts,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Resolve aggregate targets to identities while preserving exact matches."""

    target_locations: Dict[str, str] = {}
    target_states: Dict[str, int] = {}
    remaining = set(active_objects)
    for location, type_counts in target.items():
        for object_type, state_counts in type_counts.items():
            for state_value, count in state_counts.items():
                matching = sorted(
                    object_name
                    for object_name in remaining
                    if object_name.rsplit("_", 1)[0] == object_type
                    and current_locations.get(object_name) == location
                    and current_states.get(
                        object_name,
                        CLEAN_STATE,
                    ) == state_value
                )
                candidates = matching + sorted(
                    object_name
                    for object_name in remaining
                    if object_name not in matching
                    and object_name.rsplit("_", 1)[0] == object_type
                )
                if len(candidates) < count:
                    raise RuntimeError(
                        f"Cannot assign {count} {object_type} object(s) "
                        f"to {location}."
                    )
                for object_name in candidates[:count]:
                    target_locations[object_name] = location
                    target_states[object_name] = state_value
                    remaining.remove(object_name)
    if remaining:
        raise RuntimeError(
            f"The target assignment omits active objects: {sorted(remaining)}."
        )
    return target_locations, target_states


def _object_operations(
    object_name: str,
    current_location: str,
    current_state: int,
    target_location: str,
    target_state: int,
) -> List[PlannedOperation]:
    operations: List[PlannedOperation] = []
    if object_name in food:
        if target_state != current_state:
            raise RuntimeError(f"Food state cannot be changed for {object_name}.")
        operations.extend([
            PlannedOperation(
                TABLE_ROBOT,
                Call("take", {"name": object_name}, TABLE_ROBOT),
                f"{TABLE_ROBOT} picked up {object_name} from "
                f"{current_location}.",
            ),
            PlannedOperation(
                TABLE_ROBOT,
                Call("put", {"target": target_location}, TABLE_ROBOT),
                f"{TABLE_ROBOT} placed {object_name} in {target_location}.",
            ),
        ])
        return operations

    if object_name not in dishware:
        raise RuntimeError(f"Unsupported active object {object_name}.")
    if current_state == DIRTY_STATE:
        if current_location != "sink":
            operations.extend([
                PlannedOperation(
                    DISH_ROBOT,
                    Call("take", {"name": object_name}, DISH_ROBOT),
                    f"{DISH_ROBOT} picked up {object_name} from "
                    f"{current_location}.",
                ),
                PlannedOperation(
                    DISH_ROBOT,
                    Call("put", {"target": "sink"}, DISH_ROBOT),
                    f"{DISH_ROBOT} placed {object_name} in sink.",
                ),
            ])
        operations.append(PlannedOperation(
            DISH_ROBOT,
            Call("clean", {"name": object_name}, DISH_ROBOT),
            f"{DISH_ROBOT} cleaned {object_name} and placed it in "
            "drying_zone.",
        ))
        current_location = "drying_zone"
    elif current_location == "sink":
        operations.extend([
            PlannedOperation(
                DISH_ROBOT,
                Call("take", {"name": object_name}, DISH_ROBOT),
                f"{DISH_ROBOT} picked up {object_name} from sink.",
            ),
            PlannedOperation(
                DISH_ROBOT,
                Call("put", {"target": "drying_zone"}, DISH_ROBOT),
                f"{DISH_ROBOT} placed {object_name} in drying_zone.",
            ),
        ])
        current_location = "drying_zone"
    if target_state != CLEAN_STATE:
        raise RuntimeError(f"Dishware target must be clean for {object_name}.")
    if current_location != target_location:
        operations.extend([
            PlannedOperation(
                TABLE_ROBOT,
                Call("take", {"name": object_name}, TABLE_ROBOT),
                f"{TABLE_ROBOT} picked up {object_name} from "
                f"{current_location}.",
            ),
            PlannedOperation(
                TABLE_ROBOT,
                Call("put", {"target": target_location}, TABLE_ROBOT),
                f"{TABLE_ROBOT} placed {object_name} in {target_location}.",
            ),
        ])
    return operations


def _scan_batch(
    current_locations: Dict[str, str],
    current_states: Dict[str, int],
) -> ScheduledBatch:
    scene_locations = current_locations.copy()
    for object_name in cleaning_objects:
        scene_locations.setdefault(object_name, "table")
    feedback = "This is the position of existing objects:"
    for location in ADVANCED_LOCATIONS:
        objects = sorted(
            object_name
            for object_name, object_location in scene_locations.items()
            if object_location == location
        )
        if objects:
            feedback += f" {location} contains {', '.join(objects)}."
    dirty_objects = sorted(
        name for name, state in current_states.items()
        if state == DIRTY_STATE
    )
    if not dirty_objects:
        feedback += " There are no dirty objects."
    elif len(dirty_objects) == 1:
        feedback += f" Dirty object is {dirty_objects[0]}."
    else:
        feedback += f" Dirty objects are {', '.join(dirty_objects)}."
    return ScheduledBatch(
        operations=(PlannedOperation(
            TABLE_ROBOT,
            Call("scan_scene", {}, TABLE_ROBOT),
            feedback,
        ),),
        objects=(),
        completed_objects=(),
    )


def build_parallel_plan(
    *,
    stage_count: int,
    current_locations: Dict[str, str],
    current_states: Dict[str, int],
    target_locations: Dict[str, str],
    target_states: Dict[str, int],
) -> ParallelPlan:
    chains: List[Tuple[str, List[PlannedOperation]]] = []
    for object_name in sorted(target_locations):
        current_location = current_locations[object_name]
        current_state = current_states.get(object_name, CLEAN_STATE)
        target_location = target_locations[object_name]
        target_state = target_states[object_name]
        if current_location == target_location and current_state == target_state:
            continue
        operations = _object_operations(
            object_name,
            current_location,
            current_state,
            target_location,
            target_state,
        )
        if not operations:
            raise RuntimeError(
                f"No operation was planned for changed object {object_name}."
            )
        chains.append((object_name, operations))
    if len(chains) != stage_count:
        raise RuntimeError(
            f"Expected one progressive stage per changed object, got "
            f"{stage_count} stage(s) and {len(chains)} object plan(s)."
        )

    batches = [_scan_batch(current_locations, current_states)]
    decisions_per_stage = [1, *([0] * (stage_count - 1))]
    positions = [0] * len(chains)
    completed_count = 0
    while completed_count < len(chains):
        selected = []
        used_robots = set()
        completes_object = False
        for chain_index, (object_name, operations) in enumerate(chains):
            position = positions[chain_index]
            if position >= len(operations):
                continue
            operation = operations[position]
            operation_completes = position == len(operations) - 1
            if operation.robot in used_robots:
                continue
            if operation_completes and completes_object:
                continue
            selected.append(
                (chain_index, object_name, operation, operation_completes)
            )
            used_robots.add(operation.robot)
            completes_object = completes_object or operation_completes
        if not selected:
            raise RuntimeError("Unable to schedule remaining robot operations.")

        decisions_per_stage[completed_count] += 1
        batches.append(ScheduledBatch(
            operations=tuple(item[2] for item in selected),
            objects=tuple(item[1] for item in selected),
            completed_objects=tuple(
                item[1] for item in selected if item[3]
            ),
        ))
        for chain_index, _, _, operation_completes in selected:
            positions[chain_index] += 1
            if operation_completes:
                completed_count += 1
    decisions_per_stage[-1] += 1
    return ParallelPlan(tuple(batches), tuple(decisions_per_stage))

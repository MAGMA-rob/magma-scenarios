from copy import deepcopy
from typing import Dict, List, Sequence, Tuple

from magma_core.simulation.state import TaskState

from ..brs_attributes import OBJECT_TYPES, ZONES, sorting_objects


TypedAssignment = Dict[str, Dict[str, int]]
FrozenAssignment = Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...]
FruitMove = Tuple[str, str, str]
ExactFruitMove = Tuple[str, str, str]

TYPE_PRIORITY_KEY = "type_priority"
TYPE_PRIORITY_NEEDS_APPLICATION_KEY = "type_priority_needs_application"
INTACT_TYPE_AREA_RELATION_KEY = "intact_type_area"
DAMAGED_TYPE_AREA_RELATION_KEY = "damaged_type_area"
OBJECT_CONDITION_RELATION_KEY = "object_condition"
DAMAGED_CONDITION = "damaged"
OBJECT_AREAS_KEY = "_object_areas"

def empty_assignment() -> TypedAssignment:
    """Create an empty type-count assignment for every zone."""

    return {
        zone: {object_type: 0 for object_type in OBJECT_TYPES}
        for zone in ZONES
    }


def freeze_assignment(assignment: TypedAssignment) -> FrozenAssignment:
    return tuple(
        (zone, tuple(type_counts.items()))
        for zone, type_counts in assignment.items()
    )


def thaw_assignment(assignment: FrozenAssignment) -> TypedAssignment:
    return {
        zone: dict(type_counts)
        for zone, type_counts in assignment
    }


def object_type_from_name(object_name: str) -> str:
    object_type = object_name.rsplit("_", 1)[0]
    if object_type not in OBJECT_TYPES:
        raise ValueError(
            f"Unknown object type for {object_name!r}: {object_type!r}."
        )
    return object_type


def validate_assignment_structure(assignment: TypedAssignment) -> None:
    if set(assignment) != set(ZONES):
        raise ValueError(
            "The assignment must contain exactly these zones: "
            f"{ZONES}. Got {list(assignment)}."
        )
    for zone in ZONES:
        unknown_types = set(assignment[zone]) - set(OBJECT_TYPES)
        if unknown_types:
            raise ValueError(
                f"Unknown object types in {zone}: {sorted(unknown_types)}."
            )
        for object_type in OBJECT_TYPES:
            quantity = assignment[zone].get(object_type, 0)
            if not isinstance(quantity, int) or quantity < 0:
                raise ValueError(
                    f"Invalid quantity for {object_type} in {zone}: "
                    f"{quantity!r}."
                )


def validate_assignment(assignment: TypedAssignment) -> None:
    validate_assignment_structure(assignment)
    for object_type in OBJECT_TYPES:
        actual_count = sum(
            assignment[zone].get(object_type, 0)
            for zone in ZONES
        )
        expected_count = len(sorting_objects[object_type])
        if actual_count != expected_count:
            raise ValueError(
                f"Invalid total for {object_type}: expected "
                f"{expected_count}, got {actual_count}."
            )


def validate_condition_assignments(
    intact_assignment: TypedAssignment,
    damaged_assignment: TypedAssignment,
) -> None:
    validate_assignment_structure(intact_assignment)
    validate_assignment_structure(damaged_assignment)
    validate_assignment(combine_assignments(
        intact_assignment,
        damaged_assignment,
    ))


def condition_assignments_from_env_options(
    env_options: Dict[str, List[str]],
) -> Tuple[TypedAssignment, TypedAssignment]:
    intact_assignment = empty_assignment()
    damaged_assignment = empty_assignment()
    damaged_objects = set(env_options.get("damaged_objs", []))
    for zone in ZONES:
        for object_name in env_options.get(zone, []):
            object_type = object_type_from_name(object_name)
            target = (
                damaged_assignment
                if object_name in damaged_objects
                else intact_assignment
            )
            target[zone][object_type] += 1
    validate_condition_assignments(intact_assignment, damaged_assignment)
    return intact_assignment, damaged_assignment


def combine_assignments(
    intact_assignment: TypedAssignment,
    damaged_assignment: TypedAssignment,
) -> TypedAssignment:
    combined = empty_assignment()
    for zone in ZONES:
        for object_type in OBJECT_TYPES:
            combined[zone][object_type] = (
                intact_assignment[zone][object_type]
                + damaged_assignment[zone][object_type]
            )
    return combined


def get_intact_assignment(state: TaskState) -> TypedAssignment:
    assignment = state.relations.get(INTACT_TYPE_AREA_RELATION_KEY)
    if assignment is None:
        raise RuntimeError(
            f"Missing state.relations[{INTACT_TYPE_AREA_RELATION_KEY!r}]."
        )
    return deepcopy(assignment)


def get_damaged_assignment(state: TaskState) -> TypedAssignment:
    assignment = state.relations.get(DAMAGED_TYPE_AREA_RELATION_KEY)
    if assignment is None:
        raise RuntimeError(
            f"Missing state.relations[{DAMAGED_TYPE_AREA_RELATION_KEY!r}]."
        )
    return deepcopy(assignment)


def get_current_assignment(state: TaskState) -> TypedAssignment:
    intact_assignment = get_intact_assignment(state)
    damaged_assignment = get_damaged_assignment(state)
    validate_condition_assignments(intact_assignment, damaged_assignment)
    return combine_assignments(intact_assignment, damaged_assignment)


def set_intact_assignment(state: TaskState, target: TypedAssignment) -> None:
    damaged_assignment = get_damaged_assignment(state)
    validate_condition_assignments(target, damaged_assignment)
    state.relations[INTACT_TYPE_AREA_RELATION_KEY] = deepcopy(target)


def set_damaged_assignment(state: TaskState, target: TypedAssignment) -> None:
    intact_assignment = get_intact_assignment(state)
    validate_condition_assignments(intact_assignment, target)
    state.relations[DAMAGED_TYPE_AREA_RELATION_KEY] = deepcopy(target)


def get_damaged_objects(state: TaskState) -> List[str]:
    conditions = state.relations.get(OBJECT_CONDITION_RELATION_KEY)
    if conditions is None:
        raise RuntimeError(
            f"Missing state.relations[{OBJECT_CONDITION_RELATION_KEY!r}]."
        )
    return [
        object_name
        for object_name in sorted(conditions)
        if conditions[object_name] == DAMAGED_CONDITION
    ]


def count_total_objects(assignment: TypedAssignment) -> int:
    return sum(
        quantity
        for zone_counts in assignment.values()
        for quantity in zone_counts.values()
    )


def count_matching_objects(
    current: TypedAssignment,
    target: TypedAssignment,
) -> int:
    return sum(
        min(current.get(zone, {}).get(object_type, 0), expected_count)
        for zone, type_counts in target.items()
        for object_type, expected_count in type_counts.items()
    )


def extract_moves(
    current: TypedAssignment,
    target: TypedAssignment,
) -> List[FruitMove]:
    """Return the typed movements required to reach ``target``."""

    validate_assignment_structure(current)
    validate_assignment_structure(target)
    moves: List[FruitMove] = []
    for object_type in OBJECT_TYPES:
        current_total = sum(current[zone][object_type] for zone in ZONES)
        target_total = sum(target[zone][object_type] for zone in ZONES)
        if current_total != target_total:
            raise ValueError(
                f"The number of {object_type} objects cannot change: "
                f"{current_total} -> {target_total}."
            )
        sources: List[str] = []
        targets: List[str] = []
        for zone in ZONES:
            difference = (
                target[zone].get(object_type, 0)
                - current[zone].get(object_type, 0)
            )
            if difference < 0:
                sources.extend([zone] * -difference)
            elif difference > 0:
                targets.extend([zone] * difference)
        if len(sources) != len(targets):
            raise RuntimeError(
                f"Cannot build moves for {object_type}: "
                f"{len(sources)} sources but {len(targets)} destinations."
            )
        moves.extend(
            (object_type, source, destination)
            for source, destination in zip(sources, targets)
        )
    return moves


def apply_move(assignment: TypedAssignment, move: FruitMove) -> None:
    object_type, source, target = move
    if source not in ZONES or target not in ZONES:
        raise ValueError(f"Unknown movement zones: {source!r} -> {target!r}.")
    if source == target:
        raise ValueError("The source and target zones must be different.")
    if object_type not in OBJECT_TYPES:
        raise ValueError(f"Unknown object type: {object_type!r}.")
    if assignment[source].get(object_type, 0) <= 0:
        raise RuntimeError(
            f"Cannot move {object_type} from {source}: "
            "no matching object is available."
        )
    assignment[source][object_type] -= 1
    assignment[target][object_type] += 1


def format_zone(zone: str) -> str:
    return {
        "left_zone": "left tray",
        "mutual_zone": "mutual tray",
        "right_zone": "right tray",
    }.get(zone, zone.replace("_", " "))


def mark_priority_applied(state: TaskState) -> None:
    if state.properties.get(TYPE_PRIORITY_KEY) is not None:
        state.properties[TYPE_PRIORITY_NEEDS_APPLICATION_KEY] = False


def select_exact_moves(
    object_areas: Dict[str, str],
    damaged_objects: Sequence[str],
    moves: Sequence[FruitMove],
    damaged: bool,
) -> List[ExactFruitMove]:
    """Bind typed movements to stable object instance names."""

    damaged_names = set(damaged_objects)
    available_areas = deepcopy(object_areas)
    exact_moves: List[ExactFruitMove] = []
    for object_type, source, destination in moves:
        object_name = next(
            (
                name
                for name in sorted(available_areas)
                if available_areas[name] == source
                and object_type_from_name(name) == object_type
                and (name in damaged_names) == damaged
            ),
            None,
        )
        if object_name is None:
            condition = "damaged" if damaged else "intact"
            raise RuntimeError(
                f"No {condition} {object_type} is available in {source} "
                "for object movement planning."
            )
        exact_moves.append((object_name, source, destination))
        available_areas[object_name] = destination
    return exact_moves


def apply_exact_moves(
    object_areas: Dict[str, str],
    moves: Sequence[ExactFruitMove],
) -> Dict[str, str]:
    updated_areas = deepcopy(object_areas)
    for object_name, _, destination in moves:
        updated_areas[object_name] = destination
    return updated_areas

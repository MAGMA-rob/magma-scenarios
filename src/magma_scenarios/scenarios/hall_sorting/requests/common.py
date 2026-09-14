from collections import defaultdict
from typing import Dict, Iterable, List, Optional
import random
from magma_core.simulation.state import TaskState
from ..attributes import HALLS

TYPE_HALL_RELATION = "type_hall"
TYPE_HALL_NEEDS_APPLICATION_KEY = "type_hall_needs_application"
TYPE_HALL_PENDING_SOURCES_KEY = "type_hall_pending_sources"
PRIORITY_HALL_KEY = "priority_hall"
DELIVERY_LAYOUT_KEY = "delivery_layout"
PRIORITY_HALL_NEEDS_APPLICATION_KEY = "priority_hall_needs_application"


def get_object_type(object_name: str) -> str:
    """
    Extract the type from an instance name.

    Example:
        book_2 -> book
    """
    object_type, separator, instance_index = object_name.rpartition("_")

    if (
        not separator
        or not object_type
        or not instance_index.isdigit()
    ):
        raise ValueError(
            f"Invalid object instance name: {object_name!r}."
        )

    return object_type


def get_delivery_objects(state: TaskState) -> List[str]:
    """Return all object instances belonging to the current delivery."""
    delivery_layout = state.properties.get(DELIVERY_LAYOUT_KEY, {})
    reception_halls = state.attributes.get("reception_halls", [])

    return [
        object_name
        for hall in reception_halls
        for object_name in delivery_layout.get(hall, [])
    ]


def get_objects_by_type(state: TaskState) -> Dict[str, List[str]]:
    """Group delivered object instances by type."""
    grouped_objects: Dict[str, List[str]] = defaultdict(list)

    for object_name in get_delivery_objects(state):
        grouped_objects[get_object_type(object_name)].append(object_name)

    return dict(grouped_objects)


def get_available_delivery_types(state: TaskState) -> List[str]:
    """Return object types that are physically present in the delivery."""
    known_types = state.attributes.get("object_classes", [])

    objects_by_type = get_objects_by_type(state)

    return [
        object_type
        for object_type in known_types
        if objects_by_type.get(object_type)
    ]


def get_assigned_delivery_types(state: TaskState) -> List[str]:
    """
    Return delivered types having a valid storage-hall assignment.
    """
    storage_halls = set(
        state.attributes.get("storage_halls", [])
    )

    type_hall = state.relations.get(
        TYPE_HALL_RELATION,
        {},
    )

    return [
        object_type
        for object_type in get_available_delivery_types(state)
        if type_hall.get(object_type) in storage_halls
    ]


def build_typed_assignment(
    state: TaskState,
    selected_types: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Build the assignment expected by TypedPlacementStage.

    Only object types selected for the current request are counted.
    """
    storage_halls = state.attributes.get(
        "storage_halls",
        [],
    )

    known_types = state.attributes.get(
        "object_classes",
        [],
    )

    type_hall = state.relations.get(
        TYPE_HALL_RELATION,
        {},
    )

    objects_by_type = get_objects_by_type(state)

    if selected_types is None:
        selected = get_available_delivery_types(state)
    else:
        selected = list(dict.fromkeys(selected_types))

    unknown_types = [
        object_type
        for object_type in selected
        if object_type not in known_types
    ]

    if unknown_types:
        raise ValueError(
            f"Unknown object types: {unknown_types}."
        )

    assignment = {
        hall: {
            object_type: 0
            for object_type in known_types
        }
        for hall in storage_halls
    }

    for object_type in selected:
        target_hall = type_hall.get(object_type)

        if target_hall not in storage_halls:
            raise RuntimeError(
                f"No valid storage hall is assigned to "
                f"{object_type!r}."
            )

        assignment[target_hall][object_type] = len(
            objects_by_type.get(object_type, [])
        )

    return assignment


def assignment_object_count(assignment: Dict[str, Dict[str, int]]) -> int:
    """Return the total number of requested object placements."""
    return sum(
        count
        for type_counts in assignment.values()
        for count in type_counts.values()
    )

def mark_type_rules_applied(state: TaskState, applied_types: Iterable[str]) -> None:
    """
    Mark selected type-to-hall rules as applied by a delivery cycle.

    If some recently modified types were not included in the cycle,
    they remain pending.
    """
    applied = set(applied_types)

    pending_types = state.properties.get(
        TYPE_HALL_PENDING_SOURCES_KEY,
        [],
    )

    remaining_types = [
        object_type
        for object_type in pending_types
        if object_type not in applied
    ]

    if remaining_types:
        state.properties[
            TYPE_HALL_PENDING_SOURCES_KEY
        ] = remaining_types

        state.properties[
            TYPE_HALL_NEEDS_APPLICATION_KEY
        ] = True

    else:
        state.properties.pop(
            TYPE_HALL_PENDING_SOURCES_KEY,
            None,
        )

        state.properties[
            TYPE_HALL_NEEDS_APPLICATION_KEY
        ] = False


def sample_next_delivery_layout(
    current_layout: Dict[str, List[str]],
    reception_halls: List[str],
    hall_capacity: int = 4,
) -> Dict[str, List[str]]:

    if len(reception_halls) != 2:
        raise ValueError("Exactly two reception halls are required.")

    objects = [
        object_name
        for hall_objects in current_layout.values()
        for object_name in hall_objects
    ]

    if len(objects) != len(set(objects)):
        raise ValueError("The current delivery layout contains duplicate objects.")

    if not objects:
        raise ValueError("Cannot sample a new empty delivery.")

    if len(objects) > len(reception_halls) * hall_capacity:
        raise ValueError("The reception halls do not have enough capacity.")

    random.shuffle(objects)

    minimum_first_count = max(1, len(objects) - hall_capacity)

    maximum_first_count = min(hall_capacity, len(objects) - 1)

    first_count = random.randint(minimum_first_count, maximum_first_count)

    new_layout = {hall: [] for hall in HALLS}

    new_layout[reception_halls[0]] = objects[:first_count]
    new_layout[reception_halls[1]] = objects[first_count:]

    return new_layout

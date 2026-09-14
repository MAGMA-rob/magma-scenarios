import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from magma_core.domain import Call
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from copy import deepcopy
from ..hs_transitions import ResetHallDeliveryTransition
from ..hs_stages import TypedPlacementStage
from magma_core.utils.text_utils import join_with_and
from .common import (
    DELIVERY_LAYOUT_KEY,
    PRIORITY_HALL_KEY,
    PRIORITY_HALL_NEEDS_APPLICATION_KEY,
    TYPE_HALL_NEEDS_APPLICATION_KEY,
    TYPE_HALL_PENDING_SOURCES_KEY,
    TYPE_HALL_RELATION,
    assignment_object_count,
    build_typed_assignment,
    get_assigned_delivery_types,
    get_available_delivery_types,
    get_object_type,
    mark_type_rules_applied,
    sample_next_delivery_layout,
)


TYPE_CYCLE_WEIGHT = 4.0
TYPE_CYCLE_PENDING_WEIGHT = 8.0
DELIVERY_CYCLE_WEIGHT = 3.0
DELIVERY_CYCLE_PENDING_WEIGHT = 6.0

ROBOT_HOME_HALL = {
    "robot1": "Hall4",
    "robot2": "Hall2",
}
FrozenLayout = Tuple[Tuple[str, Tuple[str, ...]], ...]
FrozenTypedAssignment = Tuple[
    Tuple[str, Tuple[Tuple[str, int], ...]], ...
]


def freeze_layout(layout: Dict[str, List[str]]) -> FrozenLayout:
    return tuple((hall, tuple(objects)) for hall, objects in layout.items())


def thaw_layout(layout: FrozenLayout) -> Dict[str, List[str]]:
    return {hall: list(objects) for hall, objects in layout}


def freeze_typed_assignment(
    assignment: Dict[str, Dict[str, int]],
) -> FrozenTypedAssignment:
    return tuple(
        (hall, tuple(type_counts.items()))
        for hall, type_counts in assignment.items()
    )


def thaw_typed_assignment(
    assignment: FrozenTypedAssignment,
) -> Dict[str, Dict[str, int]]:
    return {hall: dict(type_counts) for hall, type_counts in assignment}


@dataclass(frozen=True)
class DeliveryCycleParameters:
    next_delivery_layout: FrozenLayout
    selected_types: Tuple[str, ...]
    assignment: FrozenTypedAssignment
    instruction: str
    priority_hall: Optional[str] = None
    applied_priority: bool = False
    object_targets: Tuple[Tuple[str, str], ...] = ()
    detection_halls: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannedHallOperation:
    call: Call
    feedback: str
    hall: str
    completes_object: bool = False


def _move_operation(robot: str, hall: str) -> PlannedHallOperation:
    return PlannedHallOperation(
        call=Call("move_to", {"hall": hall}, robot),
        feedback=f"{robot} moved to {hall} with its tray.",
        hall=hall,
    )


def _detect_operation(
    robot: str,
    hall: str,
) -> PlannedHallOperation:
    return PlannedHallOperation(
        call=Call("detect", {}, robot),
        feedback=f"{robot} inspected {hall}.",
        hall=hall,
    )


def _direct_transport_operations(
    robot: str,
    source_hall: str,
    target_hall: str,
    objects: Sequence[str],
    detect_source: bool = True,
) -> List[PlannedHallOperation]:
    operations = [_move_operation(robot, source_hall)]
    if detect_source:
        operations.append(_detect_operation(robot, source_hall))

    operations.extend(
        PlannedHallOperation(
            call=Call("pick_to_tray", {"object": object_name}, robot),
            feedback=f"{object_name} is now on {robot}'s tray.",
            hall=source_hall,
        )
        for object_name in objects
    )
    operations.append(_move_operation(robot, target_hall))
    operations.extend(
        PlannedHallOperation(
            call=Call("drop_from_tray", {"object": object_name}, robot),
            feedback=f"{object_name} was dropped on {target_hall}.",
            hall=target_hall,
            completes_object=True,
        )
        for object_name in objects
    )

    if target_hall != ROBOT_HOME_HALL[robot]:
        operations.append(
            _move_operation(robot, ROBOT_HOME_HALL[robot])
        )

    return operations


def build_delivery_stages(
    assignment: Dict[str, Dict[str, int]], 
    instruction: str, start_n: int = 1, 
    goal_description_builder: Optional[Callable[[int], str]] = None,
    ) -> List[TypedPlacementStage]:

    """
    Build progressive TypedPlacementStage instances.

    The final stage resets the physical environment to recreate the
    reception layout for the next request.
    """
    total_objects = assignment_object_count(assignment)

    if total_objects <= 0:
        raise ValueError("Cannot build delivery stages from an empty assignment.")

    if start_n < 1 or start_n > total_objects:
        raise ValueError(
            f"start_n must be between 1 and {total_objects}, got {start_n}."
        )

    return [
        TypedPlacementStage(
            n=minimum,
            assignment=assignment,
            instruction=(
                instruction
                if minimum == start_n
                else "none"
            ),
            goal_description=(
                goal_description_builder(minimum)
                if goal_description_builder is not None
                else None
            ),
        )
        for minimum in range(
            start_n,
            total_objects + 1,
        )
    ]

def build_priority_assignment(state: TaskState, priority_hall: str,) -> Dict[str, Dict[str, int]]:
    reception_halls = state.attributes.get("reception_halls", [])

    storage_halls = state.attributes.get("storage_halls", [])

    known_types = state.attributes.get("object_classes", [])

    if priority_hall not in storage_halls:
        raise ValueError(f"{priority_hall!r} is not a storage hall.")

    delivery_layout = state.properties.get(DELIVERY_LAYOUT_KEY, {})

    type_hall = state.relations.get(TYPE_HALL_RELATION, {})

    assignment = {
        hall: {
            object_type: 0
            for object_type in known_types
        }
        for hall in [
            *reception_halls,
            *storage_halls,
        ]
    }

    for reception_hall in reception_halls:
        for object_name in delivery_layout.get(reception_hall, []):
            object_type = get_object_type(object_name)

            if type_hall.get(object_type) == priority_hall:
                target_hall = priority_hall
            else:
                target_hall = reception_hall

            assignment[target_hall][object_type] += 1

    return assignment

def _pending_assigned_types(state: TaskState) -> List[str]:
    """
    Return recently modified types which are present in the delivery
    and have a valid storage-hall assignment.
    """
    assigned_types = set(
        get_assigned_delivery_types(state)
    )

    return [
        object_type
        for object_type in state.properties.get(
            TYPE_HALL_PENDING_SOURCES_KEY,
            [],
        )
        if object_type in assigned_types
    ]


class SortTypeCycleRequest(BaseRequest[DeliveryCycleParameters]):
    """
    Sort every delivered instance of one selected object type.
    """

    def __init__(
        self,
        cycle_weight: float = TYPE_CYCLE_WEIGHT,
        pending_rule_weight: float = TYPE_CYCLE_PENDING_WEIGHT,
    ) -> None:
        super().__init__()

        if cycle_weight < 0:
            raise ValueError(
                "cycle_weight must be non-negative."
            )

        if pending_rule_weight < 0:
            raise ValueError(
                "pending_rule_weight must be non-negative."
            )


        self.cycle_weight = cycle_weight
        self.pending_rule_weight = pending_rule_weight

    def sampling_weight(self, state: TaskState) -> float:
        assigned_types = get_assigned_delivery_types(state)

        if not assigned_types:
            return 0

        has_pending_rule = state.properties.get(
            TYPE_HALL_NEEDS_APPLICATION_KEY,
            False,
        )

        if (
            has_pending_rule
            and _pending_assigned_types(state)
        ):
            return self.pending_rule_weight

        return self.cycle_weight

    def sample_parameters(self, state: TaskState) -> DeliveryCycleParameters:
        current_layout = state.properties.get(DELIVERY_LAYOUT_KEY, {})

        reception_halls = state.attributes.get("reception_halls", [])

        next_delivery_layout = sample_next_delivery_layout(
            current_layout=current_layout,
            reception_halls=reception_halls,
        )

        delivery_state = deepcopy(state)

        delivery_state.properties[DELIVERY_LAYOUT_KEY] = deepcopy(next_delivery_layout)

        candidate_types = _pending_assigned_types(
            delivery_state
        )

        if not candidate_types:
            candidate_types = get_assigned_delivery_types(
                delivery_state
            )

        if not candidate_types:
            raise RuntimeError(
                "No delivered object type has a valid storage-hall assignment."
            )

        selected_type = random.choice(candidate_types)

        assignment = build_typed_assignment(
            state=delivery_state,
            selected_types=[selected_type],
        )

        instruction = (
            f"A delivery has arrived. Sort all "
            f"{selected_type} objects into their assigned hall."
        )

        return DeliveryCycleParameters(
            freeze_layout(next_delivery_layout),
            (selected_type,),
            freeze_typed_assignment(assignment),
            instruction,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: DeliveryCycleParameters,
    ) -> List[BaseTaskStage]:
        assignment = thaw_typed_assignment(parameters.assignment)
        stages = build_delivery_stages(assignment, parameters.instruction)

        stages[0].entry_transition = (
            ResetHallDeliveryTransition(
                delivery_layout=thaw_layout(parameters.next_delivery_layout),
            )
        )

        stages[-1].global_parameters.reset_at_end = False

        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: DeliveryCycleParameters,
    ) -> TaskState:
        state.properties[DELIVERY_LAYOUT_KEY] = thaw_layout(
            parameters.next_delivery_layout
        )

        mark_type_rules_applied(
            state=state,
            applied_types=parameters.selected_types,
        )

        return state


class SmallSortTypeCycleRequest(SortTypeCycleRequest):
    def __init__(
        self,
        max_objects: int = 2,
        cycle_weight: float = TYPE_CYCLE_WEIGHT,
        pending_rule_weight: float = TYPE_CYCLE_PENDING_WEIGHT,
    ) -> None:
        if max_objects < 1:
            raise ValueError("max_objects must be at least 1.")
        super().__init__(
            cycle_weight=cycle_weight,
            pending_rule_weight=pending_rule_weight,
        )
        self.max_objects = max_objects

    def sample_parameters(self, state: TaskState) -> DeliveryCycleParameters:
        parameters = super().sample_parameters(state)
        selected_type = parameters.selected_types[0]
        next_delivery_layout = thaw_layout(parameters.next_delivery_layout)
        candidates = [
            object_name
            for reception_hall in state.attributes.get("reception_halls", [])
            for object_name in next_delivery_layout.get(reception_hall, [])
            if get_object_type(object_name) == selected_type
        ]
        selected_count = random.randint(
            1,
            min(self.max_objects, len(candidates)),
        )
        selected_objects = random.sample(candidates, k=selected_count)
        target_hall = state.relations[TYPE_HALL_RELATION][selected_type]
        assignment = thaw_typed_assignment(parameters.assignment)
        for type_counts in assignment.values():
            type_counts[selected_type] = 0
        assignment[target_hall][selected_type] = selected_count
        object_word = "object" if selected_count == 1 else "objects"
        destination_pronoun = "its" if selected_count == 1 else "their"
        instruction = (
            "A delivery has arrived. Sort "
            f"{selected_count} {selected_type} {object_word} "
            f"into {destination_pronoun} assigned hall."
        )
        return DeliveryCycleParameters(
            parameters.next_delivery_layout,
            parameters.selected_types,
            freeze_typed_assignment(assignment),
            instruction,
            object_targets=tuple(
                (object_name, target_hall)
                for object_name in selected_objects
            ),
        )


class SortDeliveryCycleRequest(BaseRequest[DeliveryCycleParameters]):
    """
    Sort the complete delivery using all current type-to-hall rules.
    """

    def __init__(
        self,
        cycle_weight: float = DELIVERY_CYCLE_WEIGHT,
        pending_rule_weight: float = DELIVERY_CYCLE_PENDING_WEIGHT
    ) -> None:
        super().__init__()

        if cycle_weight < 0:
            raise ValueError(
                "cycle_weight must be non-negative."
            )

        if pending_rule_weight < 0:
            raise ValueError(
                "pending_rule_weight must be non-negative."
            )

        self.cycle_weight = cycle_weight
        self.pending_rule_weight = pending_rule_weight

    def sampling_weight(self, state: TaskState) -> float:
        available_types = set(get_available_delivery_types(state))

        assigned_types = set(get_assigned_delivery_types(state))

        # The complete cycle is possible only when every type in the
        # current delivery has a valid destination.
        if (
            not available_types
            or available_types != assigned_types
        ):
            return 0

        if state.properties.get(TYPE_HALL_NEEDS_APPLICATION_KEY, False):
            return self.pending_rule_weight

        return self.cycle_weight

    def sample_parameters(self, state: TaskState) -> DeliveryCycleParameters:
        current_layout = state.properties.get(DELIVERY_LAYOUT_KEY, {})
        reception_halls = state.attributes.get("reception_halls", [])
        next_delivery_layout = sample_next_delivery_layout(
            current_layout=current_layout,
            reception_halls=reception_halls,
        )
        delivery_state = deepcopy(state)
        delivery_state.properties[DELIVERY_LAYOUT_KEY] = deepcopy(
            next_delivery_layout
        )
        available_types = get_available_delivery_types(delivery_state)
        assigned_types = set(get_assigned_delivery_types(delivery_state))
        missing_types = [
            object_type
            for object_type in available_types
            if object_type not in assigned_types
        ]
        if missing_types:
            raise RuntimeError(
                "Cannot sort the complete delivery because "
                "these types have no valid storage-hall "
                f"assignment: {missing_types}."
            )
        if not available_types:
            raise RuntimeError("The current delivery contains no object.")
        assignment = build_typed_assignment(
            state=delivery_state,
            selected_types=available_types,
        )
        priority_hall = delivery_state.properties.get(PRIORITY_HALL_KEY)
        instruction = (
            "A delivery has arrived. Sort every item "
            "into its assigned hall."
        )
        applied_priority = False
        if priority_hall is not None:
            type_hall = delivery_state.relations.get(TYPE_HALL_RELATION, {})
            priority_objects = [
                object_name
                for reception_hall in reception_halls
                for object_name in next_delivery_layout.get(
                    reception_hall,  [],
                )
                if type_hall.get(get_object_type(object_name)) == priority_hall
            ]
            applied_priority = bool(priority_objects)
        return DeliveryCycleParameters(
            freeze_layout(next_delivery_layout),
            tuple(available_types),
            freeze_typed_assignment(assignment),
            instruction,
            priority_hall if applied_priority else None,
            applied_priority,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: DeliveryCycleParameters,
    ) -> List[BaseTaskStage]:
        assignment = thaw_typed_assignment(parameters.assignment)
        next_delivery_layout = thaw_layout(parameters.next_delivery_layout)
        if parameters.priority_hall is None:
            stages = build_delivery_stages(assignment, parameters.instruction)
        else:
            delivery_state = state.clone()
            delivery_state.properties[DELIVERY_LAYOUT_KEY] = next_delivery_layout
            priority_assignment = build_priority_assignment(
                delivery_state,
                parameters.priority_hall,
            )
            total_objects = assignment_object_count(assignment)
            priority_count = assignment_object_count(priority_assignment)
            if priority_count <= 0:
                stages = build_delivery_stages(assignment, parameters.instruction)
            else:
                priority_parts = [
                    (
                        f"{count} {object_type}"
                        if count == 1 else f"{count} {object_type}s"
                    )
                    for object_type, count
                    in assignment[parameters.priority_hall].items() if count > 0
                ]
                priority_description = join_with_and(priority_parts)
                non_priority_count = total_objects - priority_count
                stages = build_delivery_stages(
                    assignment=priority_assignment,
                    instruction=parameters.instruction,
                    start_n=non_priority_count + 1,
                    goal_description_builder=lambda minimum: (
                        f"Place at least {minimum - non_priority_count} "
                        f"of the priority objects "
                        f"({priority_description}) in {parameters.priority_hall}, "
                        "while leaving all other objects in reception."
                    ),
                )
                if priority_count < total_objects:
                    stages.extend(
                        build_delivery_stages(
                            assignment=assignment,
                            instruction="none",
                            start_n=priority_count + 1,
                        )
                    )
        stages[0].entry_transition = ResetHallDeliveryTransition(
            delivery_layout=next_delivery_layout,
        )
        stages[-1].global_parameters.reset_at_end = False
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: DeliveryCycleParameters,
    ) -> TaskState:
        state.properties[DELIVERY_LAYOUT_KEY] = thaw_layout(
            parameters.next_delivery_layout
        )
        mark_type_rules_applied(
            state=state,
            applied_types=parameters.selected_types,
        )
        if parameters.applied_priority:
            state.properties[PRIORITY_HALL_NEEDS_APPLICATION_KEY] = False
        return state

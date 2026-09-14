import copy
import random
from dataclasses import dataclass
from typing import List, Optional
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.utils.text_utils import join_with_and
from ..brs_attributes import OBJECT_TYPES, ZONES, sorting_objects
from ..brs_stages import RecipeObjectsStage
from .common import (
    OBJECT_AREAS_KEY,
    TYPE_PRIORITY_KEY,
    TYPE_PRIORITY_NEEDS_APPLICATION_KEY,
    ExactFruitMove,
    FrozenAssignment,
    TypedAssignment,
    apply_exact_moves,
    count_matching_objects,
    count_total_objects,
    empty_assignment,
    extract_moves,
    format_zone,
    mark_priority_applied,
    combine_assignments,
    get_damaged_assignment,
    get_damaged_objects,
    get_intact_assignment,
    set_damaged_assignment,
    set_intact_assignment,
    select_exact_moves,
    freeze_assignment,
    thaw_assignment,
)


@dataclass(frozen=True)
class SortAllFruitsParameters:
    intact_current: FrozenAssignment
    damaged_current: FrozenAssignment
    intact_goal: FrozenAssignment
    damaged_goal: FrozenAssignment
    damaged_objects: tuple[str, ...]
    exact_moves: tuple[ExactFruitMove, ...]
    resulting_object_areas: tuple[tuple[str, str], ...]
    instruction: str
    priority_type: Optional[str]


class SortAllFruits(BaseRequest[SortAllFruitsParameters]):
    """
    Ask the robots to sort every fruit type into a different zone.

    The mapping between fruit types and zones is sampled for every
    request.
    """

    def __init__(self, weight: float = 2.0) -> None:
        super().__init__()

        self.weight = weight

    @staticmethod
    def _split_goal_assignment(
        goal: TypedAssignment,
        intact_current: TypedAssignment,
        damaged_current: TypedAssignment,
    ) -> tuple[TypedAssignment, TypedAssignment]:
        intact_goal = empty_assignment()
        damaged_goal = empty_assignment()

        for object_type in OBJECT_TYPES:
            target_zone = next(
                zone
                for zone in ZONES
                if goal[zone][object_type] > 0
            )

            intact_goal[target_zone][object_type] = sum(
                intact_current[zone][object_type]
                for zone in ZONES
            )
            damaged_goal[target_zone][object_type] = sum(
                damaged_current[zone][object_type]
                for zone in ZONES
            )

        return intact_goal, damaged_goal


    def sampling_weight(self, state: TaskState) -> float:
        weight = self.weight

        if state.properties.get(TYPE_PRIORITY_NEEDS_APPLICATION_KEY, False):
            weight *= 2.0

        return weight

    def _build_goal_assignment(self, current: TypedAssignment) -> TypedAssignment:
        """
        Assign each fruit type to a different randomly selected zone.

        If the sampled assignment already matches the current world,
        another mapping is sampled.
        """

        possible_goals: List[TypedAssignment] = []

        for zone_order in self._zone_permutations():
            goal = empty_assignment()

            for object_type, zone in zip(OBJECT_TYPES, zone_order):
                goal[zone][object_type] = len(
                    sorting_objects[object_type]
                )

            if (
                count_matching_objects(current, goal)
                < count_total_objects(goal)
            ):
                possible_goals.append(goal)

        if not possible_goals:
            raise RuntimeError("No different sorting assignment can be created.")

        return random.choice(possible_goals)

    @staticmethod
    def _zone_permutations() -> List[List[str]]:
        """Return every possible one-to-one type-to-zone mapping."""

        from itertools import permutations

        return [
            list(zone_order)
            for zone_order in permutations(
                ZONES,
                len(OBJECT_TYPES),
            )
        ]

    def _build_instruction(self, goal: TypedAssignment) -> str:
        descriptions = []

        for zone in ZONES:
            assigned_types = [
                object_type
                for object_type in OBJECT_TYPES
                if goal[zone].get(object_type, 0) > 0
            ]

            for object_type in assigned_types:
                descriptions.append(
                    f"place all {object_type} objects "
                    f"in the {format_zone(zone)}"
                )

        return f"Sort all fruits by type: {join_with_and(descriptions)}."


    @staticmethod
    def _priority_assignment(goal: TypedAssignment, priority_type: str) -> TypedAssignment:
        """
        Keep only the expected positions of the priority type.

        Other types have a target count of zero and therefore do not
        participate in this intermediate goal.
        """

        assignment = empty_assignment()

        for zone in ZONES:
            assignment[zone][priority_type] = (
                goal[zone].get(priority_type, 0)
            )

        return assignment

    def _build_stages(
        self,
        current: TypedAssignment,
        goal: TypedAssignment,
        intact_goal: TypedAssignment,
        damaged_goal: TypedAssignment,
        damaged_objects: List[str],
        instruction: str,
        priority_type: Optional[str],
    ) -> List[BaseTaskStage]:

        total_objects = count_total_objects(goal)
        correct_objects = count_matching_objects(current, goal)
        total_requested_objects = total_objects - correct_objects
        completed_priority_objects = 0

        stages: List[BaseTaskStage] = []

        if priority_type is not None:
            priority_assignment = self._priority_assignment(goal, priority_type)

            priority_intact_goal = self._priority_assignment(
                intact_goal,
                priority_type,
            )
            priority_damaged_goal = self._priority_assignment(
                damaged_goal,
                priority_type,
            )

            total_priority_objects = count_total_objects(priority_assignment)
            correct_priority_objects = count_matching_objects(current, priority_assignment)
            requested_priority_objects = (
                total_priority_objects - correct_priority_objects
            )

            for minimum in range(
                correct_priority_objects + 1,
                total_priority_objects + 1,
            ):
                stages.append(
                    RecipeObjectsStage(
                        n=minimum,
                        progress=minimum - correct_priority_objects,
                        progress_total=total_requested_objects,
                        intact_assignment=copy.deepcopy(priority_intact_goal),
                        damaged_assignment=copy.deepcopy(priority_damaged_goal),
                        damaged_objects=damaged_objects.copy(),
                        instruction=instruction,
                    )
                )
                instruction = "none"

            completed_priority_objects = requested_priority_objects

            # After completing the priority stages, all priority
            # objects are considered correctly placed.
            correct_objects = (
                correct_objects
                - correct_priority_objects
                + total_priority_objects
            )

        for minimum in range(
            correct_objects + 1,
            total_objects + 1,
        ):
            stages.append(
                RecipeObjectsStage(
                    n=minimum,
                    progress=(
                        completed_priority_objects
                        + minimum
                        - correct_objects
                    ),
                    progress_total=total_requested_objects,
                    intact_assignment=copy.deepcopy(intact_goal),
                    damaged_assignment=copy.deepcopy(damaged_goal),
                    damaged_objects=damaged_objects.copy(),
                    instruction=instruction,
                ))
            instruction = "none"

        if stages:
            stages[-1].stage_input.flag_answer_to_user = True

        return stages

    def sample_parameters(self, state: TaskState) -> SortAllFruitsParameters:
        intact_current = get_intact_assignment(state)
        damaged_current = get_damaged_assignment(state)

        current = combine_assignments(
            intact_current,
            damaged_current,
        )
        goal = self._build_goal_assignment(current)

        intact_goal, damaged_goal = self._split_goal_assignment(
            goal=goal,
            intact_current=intact_current,
            damaged_current=damaged_current,
        )

        damaged_objects = get_damaged_objects(state)

        priority_type = state.properties.get(TYPE_PRIORITY_KEY)

        if priority_type is not None and priority_type not in OBJECT_TYPES:
            raise ValueError(f"Unknown priority type: {priority_type!r}.")

        instruction = self._build_instruction(goal)

        intact_moves = extract_moves(intact_current, intact_goal)
        damaged_moves = extract_moves(damaged_current, damaged_goal)
        object_areas = state.properties[OBJECT_AREAS_KEY]
        exact_moves = select_exact_moves(
            object_areas,
            damaged_objects,
            intact_moves,
            damaged=False,
        )
        exact_moves.extend(
            select_exact_moves(
                object_areas,
                damaged_objects,
                damaged_moves,
                damaged=True,
            )
        )
        if priority_type is None:
            random.shuffle(exact_moves)
        else:
            exact_moves.sort(
                key=lambda move: not move[0].startswith(f"{priority_type}_")
        )
        return SortAllFruitsParameters(
            intact_current=freeze_assignment(intact_current),
            damaged_current=freeze_assignment(damaged_current),
            intact_goal=freeze_assignment(intact_goal),
            damaged_goal=freeze_assignment(damaged_goal),
            damaged_objects=tuple(damaged_objects),
            exact_moves=tuple(exact_moves),
            resulting_object_areas=tuple(
                apply_exact_moves(object_areas, exact_moves).items()
            ),
            instruction=instruction,
            priority_type=priority_type,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: SortAllFruitsParameters,
    ) -> List[BaseTaskStage]:
        intact_current = thaw_assignment(parameters.intact_current)
        damaged_current = thaw_assignment(parameters.damaged_current)
        intact_goal = thaw_assignment(parameters.intact_goal)
        damaged_goal = thaw_assignment(parameters.damaged_goal)
        return self._build_stages(
            current=combine_assignments(intact_current, damaged_current),
            goal=combine_assignments(intact_goal, damaged_goal),
            intact_goal=intact_goal,
            damaged_goal=damaged_goal,
            damaged_objects=list(parameters.damaged_objects),
            instruction=parameters.instruction,
            priority_type=parameters.priority_type,
        )

    def apply_request(
        self,
        state: TaskState,
        parameters: SortAllFruitsParameters,
    ) -> TaskState:
        set_intact_assignment(state, thaw_assignment(parameters.intact_goal))
        set_damaged_assignment(state, thaw_assignment(parameters.damaged_goal))
        state.properties[OBJECT_AREAS_KEY] = dict(
            parameters.resulting_object_areas
        )

        mark_priority_applied(state)
        return state

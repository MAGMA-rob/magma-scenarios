import copy
import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.utils.text_utils import join_with_and

from ..brs_attributes import perturb_assignment
from ..brs_stages import RecipeObjectsStage
from .common import (
    OBJECT_AREAS_KEY,
    TYPE_PRIORITY_KEY,
    TYPE_PRIORITY_NEEDS_APPLICATION_KEY,
    ExactFruitMove,
    FrozenAssignment,
    FruitMove,
    apply_exact_moves,
    apply_move,
    combine_assignments,
    count_total_objects,
    extract_moves,
    format_zone,
    freeze_assignment,
    get_damaged_assignment,
    get_damaged_objects,
    get_intact_assignment,
    set_intact_assignment,
    select_exact_moves,
    thaw_assignment,
)
from .sorting import SortAllFruits


def _format_move(move: FruitMove, quantity: int = 1) -> str:
    object_type, source, destination = move
    object_word = "object" if quantity == 1 else "objects"
    return (
        f"move {quantity} {object_type} {object_word} "
        f"from the {format_zone(source)} to the {format_zone(destination)}"
    )


def _build_sequence_instruction(moves: List[FruitMove]) -> str:
    descriptions = [_format_move(move) for move in moves]
    if len(descriptions) == 1:
        return descriptions[0].capitalize() + ". Only move intact objects."
    return (
        descriptions[0].capitalize()
        + ", then "
        + ", then ".join(descriptions[1:])
        + ". Only move intact objects."
    )


def _build_set_instruction(moves: List[FruitMove]) -> str:
    descriptions = [
        _format_move(move, quantity)
        for move, quantity in Counter(moves).items()
    ]
    return (
        "Perform the following fruit movements in any order: "
        f"{join_with_and(descriptions)}. Only move intact objects."
    )


@dataclass(frozen=True)
class FruitMovementParameters:
    current: FrozenAssignment
    damaged_assignment: FrozenAssignment
    target: FrozenAssignment
    damaged_objects: Tuple[str, ...]
    moves: Tuple[FruitMove, ...]
    exact_moves: Tuple[ExactFruitMove, ...]
    resulting_object_areas: Tuple[Tuple[str, str], ...]
    instruction: str
    ordered: bool
    priority_type: str | None = None


class MoveFruitSequence(BaseRequest[FruitMovementParameters]):
    def __init__(self, max_moves: int = 3, weight: float = 3.0) -> None:
        super().__init__()
        if max_moves < 1:
            raise ValueError("max_moves must be at least 1.")
        self.max_moves = max_moves
        self.weight = weight

    def sampling_weight(self, state: TaskState) -> float:
        if state.properties.get(TYPE_PRIORITY_KEY) is not None:
            return 0
        return self.weight

    def sample_parameters(self, state: TaskState) -> FruitMovementParameters:
        current = get_intact_assignment(state)
        damaged_assignment = get_damaged_assignment(state)
        damaged_objects = get_damaged_objects(state)
        target, actual_move_count = perturb_assignment(
            assignment=current,
            max_moves=self.max_moves,
        )
        if actual_move_count == 0:
            raise RuntimeError("The perturbation did not produce any movement.")
        moves = extract_moves(current=current, target=target)
        random.shuffle(moves)
        instruction = _build_sequence_instruction(moves)
        object_areas = state.properties[OBJECT_AREAS_KEY]
        exact_moves = select_exact_moves(
            object_areas,
            damaged_objects,
            moves,
            damaged=False,
        )
        return FruitMovementParameters(
            current=freeze_assignment(current),
            damaged_assignment=freeze_assignment(damaged_assignment),
            target=freeze_assignment(target),
            damaged_objects=tuple(damaged_objects),
            moves=tuple(moves),
            exact_moves=tuple(exact_moves),
            resulting_object_areas=tuple(
                apply_exact_moves(object_areas, exact_moves).items()
            ),
            instruction=instruction,
            ordered=True,
        )

    @staticmethod
    def _build_stages(
        parameters: FruitMovementParameters,
    ) -> List[BaseTaskStage]:
        current = thaw_assignment(parameters.current)
        damaged_assignment = thaw_assignment(parameters.damaged_assignment)
        progressive_assignment = copy.deepcopy(current)
        total_objects = (
            count_total_objects(current)
            + count_total_objects(damaged_assignment)
        )
        stages = []
        for index, move in enumerate(parameters.moves):
            apply_move(progressive_assignment, move)
            stage = RecipeObjectsStage(
                n=total_objects,
                progress=index + 1,
                progress_total=len(parameters.moves),
                intact_assignment=copy.deepcopy(progressive_assignment),
                damaged_assignment=copy.deepcopy(damaged_assignment),
                damaged_objects=list(parameters.damaged_objects),
                instruction=parameters.instruction if index == 0 else "none",
            )
            stage.stage_input.flag_answer_to_user = False
            stage.global_parameters.reset_at_end = False
            stages.append(stage)
        if stages:
            stages[-1].stage_input.flag_answer_to_user = True
        return stages

    def create_stages(
        self,
        state: TaskState,
        parameters: FruitMovementParameters,
    ) -> List[BaseTaskStage]:
        return self._build_stages(parameters)

    def apply_request(
        self,
        state: TaskState,
        parameters: FruitMovementParameters,
    ) -> TaskState:
        set_intact_assignment(state, thaw_assignment(parameters.target))
        state.properties[OBJECT_AREAS_KEY] = dict(
            parameters.resulting_object_areas
        )
        return state


class MoveFruitSet(MoveFruitSequence):
    def sampling_weight(self, state: TaskState) -> float:
        weight = self.weight
        if state.properties.get(TYPE_PRIORITY_NEEDS_APPLICATION_KEY, False):
            weight *= 2.0
        return weight

    def sample_parameters(self, state: TaskState) -> FruitMovementParameters:
        intact_current = get_intact_assignment(state)
        damaged_current = get_damaged_assignment(state)
        damaged_objects = get_damaged_objects(state)
        target, actual_move_count = perturb_assignment(
            assignment=intact_current,
            max_moves=self.max_moves,
        )
        if actual_move_count == 0:
            raise RuntimeError("The perturbation did not produce any movement.")
        moves = extract_moves(current=intact_current, target=target)
        random.shuffle(moves)
        instruction = _build_set_instruction(moves)
        priority_type = state.properties.get(TYPE_PRIORITY_KEY)
        if priority_type is not None:
            moves.sort(key=lambda move: move[0] != priority_type)
        object_areas = state.properties[OBJECT_AREAS_KEY]
        exact_moves = select_exact_moves(
            object_areas,
            damaged_objects,
            moves,
            damaged=False,
        )
        return FruitMovementParameters(
            current=freeze_assignment(intact_current),
            damaged_assignment=freeze_assignment(damaged_current),
            target=freeze_assignment(target),
            damaged_objects=tuple(damaged_objects),
            moves=tuple(moves),
            exact_moves=tuple(exact_moves),
            resulting_object_areas=tuple(
                apply_exact_moves(object_areas, exact_moves).items()
            ),
            instruction=instruction,
            ordered=False,
            priority_type=priority_type,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: FruitMovementParameters,
    ) -> List[BaseTaskStage]:
        intact_current = thaw_assignment(parameters.current)
        damaged_current = thaw_assignment(parameters.damaged_assignment)
        return SortAllFruits()._build_stages(
            current=combine_assignments(intact_current, damaged_current),
            goal=combine_assignments(
                thaw_assignment(parameters.target),
                damaged_current,
            ),
            intact_goal=thaw_assignment(parameters.target),
            damaged_goal=damaged_current,
            damaged_objects=list(parameters.damaged_objects),
            instruction=parameters.instruction,
            priority_type=parameters.priority_type,
        )

    def apply_request(
        self,
        state: TaskState,
        parameters: FruitMovementParameters,
    ) -> TaskState:
        state = super().apply_request(state, parameters)
        if state.properties.get(TYPE_PRIORITY_KEY) is not None:
            state.properties[TYPE_PRIORITY_NEEDS_APPLICATION_KEY] = False
        return state


class SmallMoveFruitSet(MoveFruitSet):
    def __init__(
        self,
        min_moves: int = 1,
        max_moves: int = 2,
        weight: float = 3.0,
    ) -> None:
        if min_moves < 1:
            raise ValueError("min_moves must be at least 1.")
        if min_moves > max_moves:
            raise ValueError("min_moves cannot exceed max_moves.")
        super().__init__(max_moves=max_moves, weight=weight)
        self.min_moves = min_moves

    def sample_parameters(self, state: TaskState) -> FruitMovementParameters:
        sampled_move_count = random.randint(self.min_moves, self.max_moves)
        request = MoveFruitSet(
            max_moves=sampled_move_count,
            weight=self.weight,
        )
        return request.sample_parameters(state)

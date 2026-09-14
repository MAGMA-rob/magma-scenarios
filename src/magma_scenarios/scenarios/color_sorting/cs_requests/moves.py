import copy
import random
from typing import Dict, List

from magma_core.simulation.requests import BaseRequest
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.utils.text_utils import join_with_and

from ..attributes import _perturb_assignment
from ..cs_stages import SortByColorStage
from .common import (
    ColorMoveParameters,
    CubeMove,
    build_sort_stages,
    count_total_cubes,
    extract_moves,
    format_location,
    freeze_assignment,
    get_colors,
    get_current_assignment,
    mark_constraint_applied,
    priority_color,
    thaw_assignment,
)


def _get_disorder_ratio(state: TaskState) -> float:
    assignment = state.relations.get("assignment")
    if not assignment:
        return 0.0
    colors = get_colors(state)
    total_cubes = count_total_cubes(assignment)
    if total_cubes == 0:
        return 0.0
    correct_cubes = sum(
        assignment.get(f"{color}_tray", {}).get(color, 0)
        for color in colors
    )
    return max(total_cubes - correct_cubes, 0) / total_cubes


class MoveCubeSequence(BaseRequest[ColorMoveParameters]):
    def __init__(self, max_cube: int = 3) -> None:
        super().__init__()
        if max_cube < 1:
            raise ValueError("max_cube must be >= 1")
        self.max_cube = max_cube

    def sampling_weight(self, state: TaskState) -> float:
        if not state.relations.get("assignment"):
            return 0
        if state.properties.get("constraint_order") is not None:
            return 0
        return 1.0 + 9.0 * (1.0 - _get_disorder_ratio(state))

    def sample_parameters(self, state: TaskState) -> ColorMoveParameters:
        current = get_current_assignment(state)
        target, actual_move_count = _perturb_assignment(
            current,
            max_misplaced=self.max_cube,
        )
        if actual_move_count == 0:
            raise RuntimeError("The perturbation did not produce any move.")
        moves = extract_moves(current, target)
        random.shuffle(moves)
        instruction_steps = []
        for index, (color, source, destination) in enumerate(moves):
            prefix = (
                "First"
                if index == 0
                else "Finally" if index == len(moves) - 1 else "Then"
            )
            instruction_steps.append(
                f"{prefix}, move one {color} cube from the "
                f"{format_location(source)} to the "
                f"{format_location(destination)}"
            )
        return ColorMoveParameters(
            current=freeze_assignment(current),
            target=freeze_assignment(target),
            moves=tuple(moves),
            instruction=(
                "Complete these moves in this exact order: "
                + ". ".join(instruction_steps)
                + "."
            ),
            tool_mode=state.attributes.get("tool_mode", "instance"),
            ordered_todos=True,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: ColorMoveParameters,
    ) -> List[BaseTaskStage]:
        current = thaw_assignment(parameters.current)
        progressive_assignment = copy.deepcopy(current)
        total_cubes = count_total_cubes(current)
        stages: List[BaseTaskStage] = []
        for index, (color, source, destination) in enumerate(parameters.moves):
            progressive_assignment[source][color] -= 1
            progressive_assignment[destination][color] += 1
            stages.append(SortByColorStage(
                n=total_cubes,
                assignment=copy.deepcopy(progressive_assignment),
                instruction=parameters.instruction if index == 0 else "none",
                last=index == len(parameters.moves) - 1,
            ))
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: ColorMoveParameters,
    ) -> TaskState:
        state.relations["assignment"] = thaw_assignment(parameters.target)
        return state


class MoveCubeSet(MoveCubeSequence):
    def sampling_weight(self, state: TaskState) -> float:
        if not state.relations.get("assignment"):
            return 0
        weight = 1.0 + 9.0 * (1.0 - _get_disorder_ratio(state))
        if state.properties.get("constraint_order_needs_application", False):
            weight *= 2.0
        return weight

    def sample_parameters(self, state: TaskState) -> ColorMoveParameters:
        current = get_current_assignment(state)
        target, actual_move_count = _perturb_assignment(
            current,
            max_misplaced=self.max_cube,
        )
        if actual_move_count == 0:
            raise RuntimeError("The perturbation did not produce any move.")
        moves = extract_moves(current, target)
        rule = state.properties.get("constraint_order")
        ordered_color = priority_color(get_colors(state), rule)
        if ordered_color is None:
            random.shuffle(moves)
        else:
            moves.sort(key=lambda move: move[0] != ordered_color)

        grouped_moves: Dict[CubeMove, int] = {}
        for move in moves:
            grouped_moves[move] = grouped_moves.get(move, 0) + 1
        instruction_parts = []
        for (color, source, destination), count in grouped_moves.items():
            instruction_parts.append(
                f"move {count} {color} {'cube' if count == 1 else 'cubes'} "
                f"from the {format_location(source)} to the "
                f"{format_location(destination)}"
            )
        return ColorMoveParameters(
            current=freeze_assignment(current),
            target=freeze_assignment(target),
            moves=tuple(moves),
            instruction="I want you to " + join_with_and(instruction_parts) + ".",
            tool_mode=state.attributes.get("tool_mode", "instance"),
            ordered_todos=ordered_color is not None,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: ColorMoveParameters,
    ) -> List[BaseTaskStage]:
        return build_sort_stages(
            thaw_assignment(parameters.current),
            get_colors(state),
            state.properties.get("constraint_order"),
            thaw_assignment(parameters.target),
            parameters.instruction,
        )

    def apply_request(
        self,
        state: TaskState,
        parameters: ColorMoveParameters,
    ) -> TaskState:
        state = super().apply_request(state, parameters)
        mark_constraint_applied(state)
        return state

import copy
import random
from typing import List

from magma_core.simulation.requests import BaseRequest
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState

from .common import (
    ColorMoveParameters,
    SortAllColorParameters,
    build_sort_stages,
    build_sorted_target,
    count_matching_cubes,
    count_total_cubes,
    extract_moves,
    freeze_assignment,
    get_colors,
    get_current_assignment,
    mark_constraint_applied,
    priority_color,
    thaw_assignment,
)


class SortAllColor(BaseRequest[SortAllColorParameters]):
    def __init__(self, max_nb_nonsorted: int = 2) -> None:
        super().__init__()
        if max_nb_nonsorted < 0:
            raise ValueError("max_nb_nonsorted must be >= 0")
        self.max_nb_nonsorted = max_nb_nonsorted

    def sampling_weight(self, state: TaskState) -> float:
        assignment = state.relations.get("assignment")
        if not assignment:
            return 0
        colors = get_colors(state)
        total_cubes = count_total_cubes(assignment)
        if total_cubes == 0:
            return 0
        correct_cubes = sum(
            assignment.get(f"{color}_tray", {}).get(color, 0)
            for color in colors
        )
        disorder_ratio = max(total_cubes - correct_cubes, 0) / total_cubes
        weight = 1.0 + 9.0 * disorder_ratio
        if state.properties.get("constraint_order_needs_application", False):
            weight *= 2.0
        return weight

    def sample_parameters(self, state: TaskState) -> SortAllColorParameters:
        current = get_current_assignment(state)
        colors = get_colors(state)
        rule = state.properties.get("constraint_order")
        target = build_sorted_target(current, colors)
        ordered_color = priority_color(colors, rule)
        exception_colors = [
            color for color in colors if color != ordered_color
        ] or colors
        candidate_goals = []
        for candidate_color in exception_colors:
            maximum = min(
                self.max_nb_nonsorted,
                target[f"{candidate_color}_tray"][candidate_color],
            )
            for count in range(maximum + 1):
                candidate = copy.deepcopy(target)
                candidate[f"{candidate_color}_tray"][candidate_color] -= count
                candidate["table"][candidate_color] += count
                if count_matching_cubes(current, candidate) < count_total_cubes(candidate):
                    candidate_goals.append((candidate_color, count, candidate))
        if not candidate_goals:
            return SortAllColorParameters(None)

        exception_color, nonsorted_count, target = random.choice(candidate_goals)
        instruction = (
            "Please sort all the cubes by color: put each cube in the tray "
            "matching its color."
        )
        if nonsorted_count > 0:
            instruction += (
                f" Leave {nonsorted_count} {exception_color} "
                f"{'cube' if nonsorted_count == 1 else 'cubes'} on the table."
            )
        moves = extract_moves(current, target)
        if ordered_color is None:
            random.shuffle(moves)
        else:
            moves.sort(key=lambda move: move[0] != ordered_color)

        todo_colors = list(colors)
        if ordered_color in todo_colors:
            todo_colors.remove(ordered_color)
            todo_colors.insert(0, ordered_color)
        return SortAllColorParameters(
            plan=ColorMoveParameters(
                current=freeze_assignment(current),
                target=freeze_assignment(target),
                moves=tuple(moves),
                instruction=instruction,
                tool_mode=state.attributes.get("tool_mode", "instance"),
                ordered_todos=ordered_color is not None,
            ),
            todo_colors=tuple(todo_colors),
            nonsorted_color=exception_color if nonsorted_count else None,
            nonsorted_count=nonsorted_count,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: SortAllColorParameters,
    ) -> List[BaseTaskStage]:
        if parameters.plan is None:
            return []
        return build_sort_stages(
            thaw_assignment(parameters.plan.current),
            get_colors(state),
            state.properties.get("constraint_order"),
            thaw_assignment(parameters.plan.target),
            parameters.plan.instruction,
        )

    def apply_request(
        self,
        state: TaskState,
        parameters: SortAllColorParameters,
    ) -> TaskState:
        if parameters.plan is None:
            return state
        state.relations["assignment"] = thaw_assignment(parameters.plan.target)
        mark_constraint_applied(state)
        return state

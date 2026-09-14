import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState

from ..attributes import _empty_assignment
from ..cs_stages import SortByColorStage


TypedAssignment = Dict[str, Dict[str, int]]
FrozenAssignment = Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...]
CubeMove = Tuple[str, str, str]


@dataclass(frozen=True)
class ColorMoveParameters:
    current: FrozenAssignment
    target: FrozenAssignment
    moves: Tuple[CubeMove, ...]
    instruction: str
    tool_mode: str
    ordered_todos: bool


@dataclass(frozen=True)
class SortAllColorParameters:
    plan: Optional[ColorMoveParameters]
    todo_colors: Tuple[str, ...] = ()
    nonsorted_color: Optional[str] = None
    nonsorted_count: int = 0


def freeze_assignment(assignment: TypedAssignment) -> FrozenAssignment:
    return tuple(
        (location, tuple(color_counts.items()))
        for location, color_counts in assignment.items()
    )


def thaw_assignment(assignment: FrozenAssignment) -> TypedAssignment:
    return {
        location: dict(color_counts)
        for location, color_counts in assignment
    }


def get_colors(state: TaskState) -> List[str]:
    colors = state.attributes.get("known_tray_color")
    if colors is None:
        raise RuntimeError(
            "Color sorting requests require 'known_tray_color' in attributes."
        )
    if len(colors) != 2:
        raise RuntimeError(
            "Color sorting requests require exactly two colors, "
            f"got {colors!r}."
        )
    return list(colors)


def get_current_assignment(state: TaskState) -> TypedAssignment:
    assignment = state.relations.get("assignment")
    if assignment is None:
        raise RuntimeError("Missing state.relations['assignment'].")
    if "table" not in assignment:
        raise RuntimeError(
            "The color-sorting assignment must contain a 'table' zone."
        )
    return copy.deepcopy(assignment)


def extract_moves(
    current: TypedAssignment,
    target: TypedAssignment,
) -> List[CubeMove]:
    locations = list(current)
    colors = sorted({
        color
        for location_counts in current.values()
        for color in location_counts
    })
    moves: List[CubeMove] = []
    for color in colors:
        sources: List[str] = []
        destinations: List[str] = []
        for location in locations:
            difference = target[location].get(color, 0) - current[location].get(color, 0)
            if difference < 0:
                sources.extend([location] * -difference)
            elif difference > 0:
                destinations.extend([location] * difference)
        if len(sources) != len(destinations):
            raise RuntimeError(
                f"Invalid perturbation for {color}: {len(sources)} sources "
                f"and {len(destinations)} destinations."
            )
        moves.extend(
            (color, source, destination)
            for source, destination in zip(sources, destinations)
        )
    return moves


def count_total_cubes(assignment: TypedAssignment) -> int:
    return sum(
        count
        for location_counts in assignment.values()
        for count in location_counts.values()
    )


def count_matching_cubes(
    current: TypedAssignment,
    target: TypedAssignment,
) -> int:
    return sum(
        min(current.get(location, {}).get(color, 0), expected_count)
        for location, color_counts in target.items()
        for color, expected_count in color_counts.items()
    )


def mark_constraint_applied(state: TaskState) -> None:
    if state.properties.get("constraint_order") is None:
        return
    state.properties["constraint_order_needs_application"] = False
    state.properties["constraint_order_applications"] = (
        state.properties.get("constraint_order_applications", 0) + 1
    )


def priority_color(colors: Sequence[str], rule: Optional[str]) -> Optional[str]:
    if rule == "first-color":
        return colors[0]
    if rule == "second-color":
        return colors[1]
    if rule is not None:
        raise ValueError(f"Unknown constraint_order: {rule!r}.")
    return None


def format_location(location: str) -> str:
    if location == "table":
        return "table"
    if location.endswith("_tray"):
        return f"{location.removesuffix('_tray')} tray"
    return location


def build_sort_stages(
    current: TypedAssignment,
    colors: List[str],
    rule: Optional[str],
    goal: TypedAssignment,
    instruction: str,
) -> List[BaseTaskStage]:
    """Preserve the scenario's progressive stage construction."""

    total_cubes = count_total_cubes(current)
    correct_cubes = count_matching_cubes(current, goal)
    stages: List[BaseTaskStage] = []
    ordered_color = priority_color(colors, rule)
    if ordered_color is not None:
        priority_assignment = {
            location: {ordered_color: counts.get(ordered_color, 0)}
            for location, counts in goal.items()
        }
        total_priority = count_total_cubes(priority_assignment)
        correct_priority = count_matching_cubes(current, priority_assignment)
        for minimum in range(correct_priority + 1, total_priority + 1):
            stages.append(SortByColorStage(
                n=minimum,
                assignment=copy.deepcopy(priority_assignment),
                instruction=instruction,
                last=False,
            ))
            instruction = "none"
        correct_cubes = correct_cubes - correct_priority + total_priority

    for minimum in range(correct_cubes + 1, total_cubes + 1):
        stages.append(SortByColorStage(
            n=minimum,
            assignment=copy.deepcopy(goal),
            instruction=instruction,
            last=minimum == total_cubes,
        ))
        instruction = "none"
    if stages:
        stages[-1].stage_input.flag_answer_to_user = True
        stages[-1].target_tool_calls = 4
    return stages


def build_sorted_target(
    current: TypedAssignment,
    colors: Sequence[str],
) -> TypedAssignment:
    target = _empty_assignment(list(colors))
    for color in colors:
        target[f"{color}_tray"][color] = sum(
            counts.get(color, 0) for counts in current.values()
        )
    return target

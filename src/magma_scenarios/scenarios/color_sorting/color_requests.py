import random
from typing import Dict, List

from magma_core.base.stage import BaseTaskStage
from magma_core.base.state.task_state import TaskState
from magma_core.base.user_request import BaseConstraintRequest, BaseRequest
from magma_core.base.constraints import BaseConstraint
from magma_core.base.data_structures import UserInstruction, EmptyInstruction

from .color_sorting_stages import SortByColorStage, ExactSortByColorStage

constraint_type = ["alternate", "first-color", "second-color"]

def _resolve_attributes_and_colors(state: TaskState) -> tuple[Dict, List[str]]:
    attributes = state.attributes.copy()
    colors = attributes.get("known_box_color", attributes.get("box_color", None))
    if colors is None:
        raise RuntimeError("Color sorting requests require 'known_box_color' or 'box_color' in state.attributes")
    if len(colors) != 2:
        raise RuntimeError(f"Color sorting requests require exactly 2 colors, got {colors}")

    attributes["known_box_color"] = list(colors)
    return attributes, list(colors)


class FirstColorConstraint(BaseConstraint):

    def apply(self, state: TaskState):
        state.properties["constraint_order"] = "first-color"

class SecondColorConstraint(BaseConstraint):

    def apply(self, state: TaskState):
        state.properties["constraint_order"] = "second-color"


class AlternateConstraint(BaseConstraint):

    def apply(self, state: TaskState):
        state.properties["constraint_order"] = "alternate"


class GiveOrderConstraint(BaseConstraintRequest):

    def __init__(self):
        super().__init__()

    def initialize_constraints(self, state: TaskState):
        _, colors = _resolve_attributes_and_colors(state)
        t = random.choice(constraint_type)
        if t == "alternate":
            self.constraints = [AlternateConstraint()]
            self.constraint_msg = (
                f"Each time I ask you to sort cubes, you must always alternate between one {colors[0]} and one {colors[1]}."
            )
        elif t == "first-color":
            self.constraints = [FirstColorConstraint()]
            self.constraint_msg = f"Each time I ask you to sort cubes, you must always sort all {colors[0]} cubes first."
        else:
            self.constraints = [SecondColorConstraint()]
            self.constraint_msg = f"Each time I ask you to sort cubes, you must always sort all {colors[1]} cubes first."


class AskForCycle(BaseRequest):

    def __init__(self, max_cube : int = 3) -> None:
        super().__init__()
        if max_cube > 6:
            raise RuntimeError("There is not enought cube inside the env. If you increased manually in the env, modify this raise.")
        self.max_cube = max_cube

    def sampling_weight(self, state: TaskState) -> float:
        return 3

    def _sample_color_counts(self, colors: List[str]) -> Dict[str, int]:
        nb = random.randint(1, self.max_cube)
        all_cubes = []
        for color in colors:
            all_cubes.extend([color] * 3)
        random.shuffle(all_cubes)

        counts = {color: 0 for color in colors}
        for cube in all_cubes[:nb]:
            counts[cube] += 1

        return counts

    def _sample_alternating_order(self, colors: List[str]) -> List[str]:
        min_nb = 2 if self.max_cube >= 2 else 1
        nb = random.randint(min_nb, self.max_cube)
        start_color = random.choice(colors)
        other_color = colors[1] if start_color == colors[0] else colors[0]

        return [
            start_color if i % 2 == 0 else other_color
            for i in range(nb)
        ]

    def _build_instruction(self, colors: List[str], counts: Dict[str, int]) -> UserInstruction:
        return UserInstruction(
            f"Please store {counts[colors[0]]} {colors[0]} cubes and {counts[colors[1]]} {colors[1]} cubes."
        )

    def _build_unordered_stages(
            self,
            instruction: UserInstruction,
            attributes: Dict,
            counts: Dict[str, int],
        ) -> List[BaseTaskStage]:
        total = sum(counts.values())
        stages = []
        cur_instruction = instruction
        for i in range(total):
            stages.append(
                SortByColorStage(
                    cur_instruction,
                    nb_good_place=i + 1,
                    attributes=attributes,
                    last=(i == total - 1),
                    max_assignment=counts
                )
            )
            cur_instruction = EmptyInstruction()

        return stages

    def _build_ordered_stages(
            self,
            instruction: UserInstruction,
            attributes: Dict,
            colors: List[str],
            color_order: List[str],
        ) -> List[BaseTaskStage]:
        stages = []
        current_instruction = instruction
        assignment = {color: 0 for color in colors}

        for i, color in enumerate(color_order):
            if color not in assignment:
                raise ValueError(f"Unknown color order entry: {color}")
            assignment[color] += 1

            stages.append(
                ExactSortByColorStage(
                    current_instruction,
                    attributes=attributes,
                    assignment=assignment.copy(),
                    last=(i == len(color_order) - 1)
                )
            )
            current_instruction = EmptyInstruction()

        return stages

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        attributes, colors = _resolve_attributes_and_colors(state)
        rule = state.properties.get("constraint_order", None)

        if rule is None:
            counts = self._sample_color_counts(colors)
            instruction = self._build_instruction(colors, counts)
            return self._build_unordered_stages(instruction, attributes, counts)

        if rule == "alternate":
            color_order = self._sample_alternating_order(colors)
        elif rule == "first-color":
            counts = self._sample_color_counts(colors)
            color_order = ([colors[0]] * counts[colors[0]]) + ([colors[1]] * counts[colors[1]])
        elif rule == "second-color":
            counts = self._sample_color_counts(colors)
            color_order = ([colors[1]] * counts[colors[1]]) + ([colors[0]] * counts[colors[0]])
        elif rule == "green-first":
            counts = self._sample_color_counts(colors)
            first_color = "green" if "green" in colors else colors[0]
            second_color = colors[1] if first_color == colors[0] else colors[0]
            color_order = ([first_color] * counts[first_color]) + ([second_color] * counts[second_color])
        elif rule == "yellow-first":
            counts = self._sample_color_counts(colors)
            first_color = "yellow" if "yellow" in colors else colors[0]
            second_color = colors[1] if first_color == colors[0] else colors[0]
            color_order = ([first_color] * counts[first_color]) + ([second_color] * counts[second_color])
        else:
            raise ValueError(f"Unknown constraint_order: {rule}")

        counts = {color: color_order.count(color) for color in colors}
        instruction = self._build_instruction(colors, counts)
        return self._build_ordered_stages(instruction, attributes, colors, color_order)

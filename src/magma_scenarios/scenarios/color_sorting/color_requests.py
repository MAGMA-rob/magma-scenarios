import random
from typing import List, Tuple

from magma_core.base.stage import BaseTaskStage
from magma_core.base.state.task_state import TaskState
from magma_core.base.user_request import BaseConstraintRequest, BaseRequest
from magma_core.base.constraints import BaseConstraint
from magma_core.base.data_structures import UserInstruction, EmptyInstruction

from .color_sorting_stages import SortByColorStage, ExactSortByColorStage

constraint_type = ["alternate", "yellow-first", "green-first"]


class GreenFirstConstraint(BaseConstraint):

    def apply(self, state: TaskState):
        state.properties["constraint_order"] = "green-first"

class YellowFirstConstraint(BaseConstraint):

    def apply(self, state: TaskState):
        state.properties["constraint_order"] = "yellow-first"


class AlternateConstraint(BaseConstraint):

    def apply(self, state: TaskState):
        state.properties["constraint_order"] = "alternate"


class GiveOrderConstraint(BaseConstraintRequest):

    def __init__(self):
        super().__init__()

    def initialize_constraints(self, state: TaskState):
        t = random.choice(constraint_type)
        if t == "alternate":
            self.constraints = [AlternateConstraint()]
            self.constraint_msg = "Each time I ask you to sort cubes, you must always alternate between one yellow and one green."
        elif t == "green-first":
            self.constraints = [GreenFirstConstraint()]
            self.constraint_msg = "Each time I ask you to sort cubes, you must always sort all green cubes first."
        else:
            self.constraints = [YellowFirstConstraint()]
            self.constraint_msg = "Each time I ask you to sort cubes, you must always sort all yellow cubes first."


class AskForCycle(BaseRequest):

    def __init__(self, max_cube : int = 3) -> None:
        super().__init__()
        if max_cube > 6:
            raise RuntimeError("There is not enought cube inside the env. If you increased manually in the env, modify this raise.")
        self.max_cube = max_cube

    def sampling_weight(self, state: TaskState) -> float:
        return 3

    def _sample_color_counts(self) -> Tuple[int, int]:
        nb = random.randint(1, self.max_cube)
        all_cubes = ["yellow", "green"] * 3
        random.shuffle(all_cubes)

        green_count = 0
        yellow_count = 0
        for cube in all_cubes[:nb]:
            if cube == "yellow":
                yellow_count += 1
            else:
                green_count += 1

        return green_count, yellow_count

    def _sample_alternating_order(self) -> List[str]:
        min_nb = 2 if self.max_cube >= 2 else 1
        nb = random.randint(min_nb, self.max_cube)
        start_color = random.choice(["yellow", "green"])
        other_color = "green" if start_color == "yellow" else "yellow"

        return [
            start_color if i % 2 == 0 else other_color
            for i in range(nb)
        ]

    def _build_instruction(self, green_count: int, yellow_count: int) -> UserInstruction:
        return UserInstruction(
            f"Please store {yellow_count} yellow cubes and {green_count} green cubes."
        )

    def _build_unordered_stages(
            self,
            instruction: UserInstruction,
            green_count: int,
            yellow_count: int,
        ) -> List[BaseTaskStage]:
        total = green_count + yellow_count
        stages = []
        cur_instruction = instruction
        for i in range(total):
            stages.append(
                SortByColorStage(
                    cur_instruction,
                    nb_good_place=i + 1,
                    last=(i == total - 1),
                    max_green=green_count,
                    max_yellow=yellow_count
                )
            )
            cur_instruction = EmptyInstruction()

        return stages

    def _build_ordered_stages(
            self,
            instruction: UserInstruction,
            color_order: List[str],
        ) -> List[BaseTaskStage]:
        stages = []
        current_instruction = instruction
        green_count = 0
        yellow_count = 0

        for i, color in enumerate(color_order):
            if color == "green":
                green_count += 1
            elif color == "yellow":
                yellow_count += 1
            else:
                raise ValueError(f"Unknown color order entry: {color}")

            stages.append(
                ExactSortByColorStage(
                    current_instruction,
                    nb_green=green_count,
                    nb_yellow=yellow_count,
                    last=(i == len(color_order) - 1)
                )
            )
            current_instruction = EmptyInstruction()

        return stages

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        rule = state.properties.get("constraint_order", None)

        if rule is None:
            green_count, yellow_count = self._sample_color_counts()
            instruction = self._build_instruction(green_count, yellow_count)
            return self._build_unordered_stages(instruction, green_count, yellow_count)

        if rule == "alternate":
            color_order = self._sample_alternating_order()
        elif rule == "green-first":
            green_count, yellow_count = self._sample_color_counts()
            color_order = (["green"] * green_count) + (["yellow"] * yellow_count)
        elif rule == "yellow-first":
            green_count, yellow_count = self._sample_color_counts()
            color_order = (["yellow"] * yellow_count) + (["green"] * green_count)
        else:
            raise ValueError(f"Unknown constraint_order: {rule}")

        green_count = color_order.count("green")
        yellow_count = color_order.count("yellow")
        instruction = self._build_instruction(green_count, yellow_count)
        return self._build_ordered_stages(instruction, color_order)

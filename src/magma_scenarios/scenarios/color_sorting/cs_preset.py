# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import copy
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
from magma_core.simulation.tasks import BaseTask, InitializationParameters
from magma_core.simulation.data_structures import SituationInit
from .cs_tool import ColorDetectionTools
from .cs_stages import SortByColorStage
from .attributes import available_colors, _build_goal_assignment, _perturb_assignment, _empty_assignment
from typing import Literal
from .cs_tool import ColorDetectionTools
from .instance_cs_tool import InstanceColorTools




class CleanTablePreset(BaseTask):
    """
    The robot must sort cubes by color. Some cubes may start on the table
    or in the wrong tray.
    """

    name: str = "Sorting Cube by Color with Detection"
    maniskill_env_id = "PartialSixCubesTwoBoxesOnTable"

    randomized_config_path = str(
        Path(__file__).resolve().parent / "detection_randomization.yaml"
    )

    def __init__(self, cubes_per_color: int = 3, max_misplaced: int = 6, tool_mode: str = "instance") -> None:
        super().__init__()

        available_tools = {
            "color": ColorDetectionTools,
            "instance": InstanceColorTools,
        }

        if tool_mode not in available_tools:
            raise ValueError(
                "tool_mode must be 'color' or 'instance', "
                f"got {tool_mode!r}"
            )

        self.Tools_cls = available_tools[tool_mode]

        if cubes_per_color < 1:
            raise ValueError(
                f"cubes_per_color must be >= 1, got {cubes_per_color}"
            )

        colors = random.sample(available_colors, k=2)

        self.goal_assignment = _build_goal_assignment(
            colors=colors,
            cubes_per_color=cubes_per_color,
        )

        initial_assignment, actual_misplaced = _perturb_assignment(
            assignment=self.goal_assignment,
            max_misplaced=max_misplaced,
        )

        self.situation_init = SituationInit(
            attributes={
                "known_tray_color": colors,
            }
        )

        self.initialization_parameters = InitializationParameters(
            env_options={
                "colors": colors,
                "composition": initial_assignment,
            }
        )

        total_targets = cubes_per_color * len(colors)

        if actual_misplaced == 0:
            start_n = total_targets
        else:
            start_n = total_targets - actual_misplaced + 1

        instruction = (
            "Sort all cubes by color. "
            "Each cube must be placed in the tray matching its color. "
            "Use the detection function to obtain information about objects."
        )

        self.stages = []

        for n in range(start_n, total_targets + 1):
            self.stages.append(
                SortByColorStage(
                    n=n,
                    assignment=self.goal_assignment,
                    instruction=instruction if n == start_n else "none",
                    last=False,
                )
            )


def _build_cross_goal_assignment(colors: List[str], cube_counts: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    color_a, color_b = colors

    count_a = cube_counts[color_a]
    count_b = cube_counts[color_b]

    balanced_count = min(count_a, count_b)

    assignment = _empty_assignment(colors)

    assignment[f"{color_a}_tray"][color_b] = balanced_count
    assignment[f"{color_b}_tray"][color_a] = balanced_count

    assignment["table"][color_a] = count_a - balanced_count
    assignment["table"][color_b] = count_b - balanced_count

    return assignment

class CrossColorPreset(BaseTask):
    name: str = "Cross Color Sorting"
    maniskill_env_id = "PartialSixCubesTwoBoxesOnTable"
    Tools_cls = ColorDetectionTools
    randomized_config_path = str(
        Path(__file__).resolve().parent / "detection_randomization.yaml"
    )

    def __init__(self, cubes_per_color: int = 3, max_misplaced: int = 2) -> None:
        super().__init__()

        if cubes_per_color < 1:
            raise ValueError(
                f"cubes_per_color must be >= 1, got {cubes_per_color}"
            )

        colors = random.sample(available_colors, k=2)

        cube_counts = {
            colors[0]: cubes_per_color,
            colors[1]: random.randint(cubes_per_color, cubes_per_color + 2),
        }

        self.goal_assignment = _build_cross_goal_assignment(
            colors=colors,
            cube_counts=cube_counts,
        )

        initial_assignment, actual_misplaced = _perturb_assignment(
            assignment=self.goal_assignment,
            max_misplaced=max_misplaced,
        )

        self.situation_init = SituationInit(
            attributes={
                "known_tray_color": colors,
            }
        )

        self.initialization_parameters = InitializationParameters(
            env_options={
                "colors": colors,
                "composition": initial_assignment,
            }
        )

        total_targets = sum(
                count
                for color_counts in self.goal_assignment.values()
                for count in color_counts.values()
            )

        if actual_misplaced == 0:
            start_n = total_targets
        else:
            start_n = total_targets - actual_misplaced + 1

        instruction = (
            "Arrange the cubes so the two trays contain matching opposite groups: "
            "each tray should contain the same number of cubes from the other tray's color. "
            "If one color has more cubes than the other, leave the unmatched extra cubes on the table."
        )

        self.stages = []

        for n in range(start_n, total_targets + 1):
            self.stages.append(
                SortByColorStage(
                    n=n,
                    assignment=self.goal_assignment,
                    instruction=instruction if n == start_n else "none",
                    last=n == total_targets,
                )
            )

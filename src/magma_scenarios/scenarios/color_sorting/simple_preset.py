# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

from magma_core.base.tasks import BaseTask, BaseBenchmarkTask
from magma_core.base.tasks_style import TaskStyle
from magma_core.base.data_structures import UserInstruction, EmptyInstruction

from .detection_tool import ColorDetectionTools
from .color_sorting_stages import SortByColorStage, DetectionStage
from .attributes import available_colors
from .color_sorting_errors import MaskRemainingCubesError, GraspCubeFailureError

class CleanTablePreset(BaseTask):
    """
    The robot must sort object by color. The scene starts with some already sorted. The robot must sort the rest.
    """

    name : str = "Sorting Cube by Color with Detection"

    env_id = "PartialSixCubesTwoBoxesOnTable"

    Tools_cls = ColorDetectionTools

    styles = [
        TaskStyle.COLOR_REASONING,
        TaskStyle.DETECTION_TASK
    ]

    randomized_config_path = str(Path(__file__).resolve().parent / "detection_randomization.yaml")

    def __init__(self, nb_already_sorted : int = 4) -> None:
        """
        To vary the difficulty, you can change the nb_already_sorted. Easy = 4-5, Medium = 2-3 Hard = 1.
        """
        super().__init__()
        
        colors = random.sample(available_colors, k=2)
        self.all_task_attributes = {
            "known_box_color" : colors
        }
        self.env_options = {
            "nb_cube_completed":nb_already_sorted,
            "colors": colors
        }
        self.tools_constant = {"colors":colors}

        self.stages = [SortByColorStage(
            instruction=UserInstruction("Clean the table please. You can use the detection function to obtain information about objects."),
            nb_good_place=nb_already_sorted+1,
            attributes = self.all_task_attributes,
            last=False
        )]
        for j in range(1, 6-nb_already_sorted):
            self.stages.append(SortByColorStage(
                instruction=EmptyInstruction(),
                nb_good_place=nb_already_sorted+j+1,
                attributes=self.all_task_attributes,
                last=False
            ))
        self.stages.append(DetectionStage(False,self.all_task_attributes,EmptyInstruction()))
    
        if nb_already_sorted < 2:
            self.approximal_difficulty = "Hard"
        elif nb_already_sorted < 4:
            self.approximal_difficulty = "Medium"
        else:
            self.approximal_difficulty = "Easy"

class ColorSortingBenchmark(BaseBenchmarkTask):
    """
    Benchmark class for color sorting scenario.
    The idea is to sort cubes of color while being robust to uncertainty (explicit observation / action failure injected)
    """
    name : str = "Benchmark Color Sorting"
    env_id : str = "PartialSixCubesTwoBoxesOnTable"

    randomized_config_path = str(Path(__file__).parent.joinpath("detection_randomization.yaml"))

    Tools_cls = ColorDetectionTools

    benchmark_possible_errors = [
        MaskRemainingCubesError(),
        GraspCubeFailureError()
    ]

    env_options = {
        "colors" : ["green","yellow"]
    }
    tools_constant = {
        "colors": ["green","yellow"]
    }
    all_task_attributes = {
        "known_box_color" : ["green","yellow"]
    }

    def _get_active_colors(self) -> List[str]:
        # scenario_variants are interpreted with a canonical benchmark order:
        # [replacement_for_green, replacement_for_yellow].
        colors = self.all_task_attributes.get("known_box_color", None)
        if colors is None:
            colors = self.tools_constant.get("colors", self.env_options.get("colors", None))

        if (
            not isinstance(colors, list)
            or len(colors) != 2
            or any(not isinstance(color, str) or color == "" for color in colors)
        ):
            raise RuntimeError(
                "ColorSortingBenchmark requires exactly two colors ordered as "
                "[replacement_for_green, replacement_for_yellow]."
            )

        colors = list(colors)

        tool_colors = self.tools_constant.get("colors", None)
        env_colors = self.env_options.get("colors", None)

        if tool_colors is not None and list(tool_colors) != colors:
            raise RuntimeError(
                "ColorSortingBenchmark requires tools_constant['colors'] to match "
                "all_task_attributes['known_box_color']."
            )
        if env_colors is not None and list(env_colors) != colors:
            raise RuntimeError(
                "ColorSortingBenchmark requires env_options['colors'] to match "
                "all_task_attributes['known_box_color']."
            )

        return colors

    def _get_rewrite_pairs(self) -> List[Tuple[str, str]]:
        green_color, yellow_color = self._get_active_colors()
        replacements = [
            ("green_box_pose", f"{green_color}_box_pose"),
            ("yellow_box_pose", f"{yellow_color}_box_pose"),
            ("green_cube", f"{green_color}_cube"),
            ("yellow_cube", f"{yellow_color}_cube"),
            ("green box", f"{green_color} box"),
            ("yellow box", f"{yellow_color} box"),
            ("green cubes", f"{green_color} cubes"),
            ("yellow cubes", f"{yellow_color} cubes"),
            ("green cube", f"{green_color} cube"),
            ("yellow cube", f"{yellow_color} cube"),
            ("green", green_color),
            ("yellow", yellow_color),
        ]
        return sorted(replacements, key=lambda item: len(item[0]), reverse=True)

    def _rewrite_string(self, value: str, replacements: List[Tuple[str, str]]) -> str:
        rewritten = value
        sentinels: List[Tuple[str, str]] = []

        for idx, (src, _) in enumerate(replacements):
            sentinel = f"__MAGMA_COLOR_SORTING_REWRITE_{idx}__"
            rewritten = rewritten.replace(src, sentinel)
            sentinels.append((sentinel, src))

        for (sentinel, _), (_, dst) in zip(sentinels, replacements):
            rewritten = rewritten.replace(sentinel, dst)

        return rewritten

    def _rewrite_value(self, value: Any, replacements: List[Tuple[str, str]]) -> Any:
        if isinstance(value, dict):
            return {key: self._rewrite_value(val, replacements) for key, val in value.items()}
        if isinstance(value, list):
            return [self._rewrite_value(item, replacements) for item in value]
        if isinstance(value, str):
            return self._rewrite_string(value, replacements)
        return value

    def rewrite_benchmark_task_payload(self, task_data: Dict) -> Dict:
        # Rewrite the canonical green/yellow benchmark JSON task so it matches
        # the currently active scene colors selected for this try.
        return self._rewrite_value(task_data, self._get_rewrite_pairs())

# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
from pathlib import Path
from typing import Any, Dict, List, Tuple
from magma_core.simulation.tasks import BaseTask, BaseBenchmarkTask, InitializationParameters
from .cs_tool import ColorDetectionTools



class ColorSortingBenchmark(BaseBenchmarkTask):
    """
    Benchmark class for color sorting scenario.
    The idea is to sort cubes of color while being robust to uncertainty (explicit observation / action failure injected)
    """
    name : str = "Benchmark Color Sorting"
    maniskill_env_id : str = "PartialSixCubesTwoBoxesOnTable"

    randomized_config_path = str(Path(__file__).parent.joinpath("detection_randomization.yaml"))

    Tools_cls = ColorDetectionTools

    benchmark_possible_errors = []

    initialization_parameters = InitializationParameters(
        env_options = {
            "colors" : ["green","yellow"]
        }
    )
    
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
            colors = self.tools_constant.get("colors", self.initialization_parameters.env_options.get("colors", None))

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

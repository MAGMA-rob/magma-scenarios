# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
from pathlib import Path
import random

from magma_core.simulation.tasks import TaskDefinition
from magma_core.simulation.state import TaskState
from magma_core.simulation.data_structures import SituationInit
from .attributes import (
    _build_goal_assignment,
    _perturb_assignment,
    available_colors,
)
from .cs_tool import ColorDetectionTools
from .instance_cs_tool import InstanceColorTools
from .cs_requests import GiveOrderConstraint, MoveCubeSequence, MoveCubeSet, SortAllColor
from .rule_renderer import ColorSortingRuleRenderer

class SortingDefinition(TaskDefinition):
    """
    The robot must sort object by color. The scene starts with some already sorted. The robot must sort the rest.
    """

    maniskill_env_id = "PartialSixCubesTwoBoxesOnTable"

    RuleRenderer_cls = ColorSortingRuleRenderer

    active_requests = [
        GiveOrderConstraint(),
        MoveCubeSequence(3),
        MoveCubeSet(3),
        SortAllColor(),
    ]

    def __init__(self, tool_mode: str = "instance"):

        available_tools = {
            "color": ColorDetectionTools,
            "instance": InstanceColorTools,
        }

        if tool_mode not in available_tools:
            raise ValueError(
                "tool_mode must be 'color' or 'instance', "
                f"got {tool_mode!r}"
            )

        selected_tools = available_tools[tool_mode]
        colors = random.sample(available_colors, k=2)
        assignment, _ = _perturb_assignment(
            _build_goal_assignment(colors, 4)
        )

        super().__init__(
            name="Sorting Cube Definition",
            situation_init=SituationInit(
                memory={},
                attributes={
                    "known_tray_color": colors,
                    "tool_mode": tool_mode,
                },
                all_task_attributes={
                    "known_tray_color": available_colors.copy(),
                    "tool_mode": tool_mode,
                },
            ),
            randomized_config_path=str(Path(__file__).resolve().parent / "detection_randomization.yaml"),
            Tools_cls=selected_tools,
        )

        self.starting_state = TaskState()
       
        self.initialization_parameters.env_options = {
            "colors": colors,
            "composition": assignment,
        }
        self.starting_state.attributes = {
            "known_tray_color": colors,
            "tool_mode": tool_mode,
        }

        self.starting_state.relations["assignment"] = assignment

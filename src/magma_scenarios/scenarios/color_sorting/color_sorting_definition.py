# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
from pathlib import Path
from typing import Dict, List

from magma_core.base.tasks import TaskDefinition
from magma_core.base.state import TaskState

from .detection_tool import ColorDetectionTools
from .color_requests import AskForCycle, GiveOrderConstraint
from .attributes import att

class SortingDefinition(TaskDefinition):
    """
    The robot must sort object by color. The scene starts with some already sorted. The robot must sort the rest.
    """

    env_id = "PartialSixCubesTwoBoxesOnTable"

    Tools_cls = ColorDetectionTools

    active_requests = [
        GiveOrderConstraint(),
        AskForCycle(3),
    ]

    def __init__(self):
        super().__init__(
            name = "Sorting Cube Definition",
            randomized_config_path=str(Path(__file__).resolve().parent / "detection_randomization.yaml")
        )

        self.starting_state = TaskState()
        self.starting_state.attributes = att

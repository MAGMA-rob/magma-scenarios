# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
from pathlib import Path

from magma_core.base.tasks import BaseTask
from magma_core.base.tasks_style import TaskStyle
from magma_core.base.data_structures import UserInstruction, EmptyInstruction

from .detection_tool import ColorDetectionTools
from .color_sorting_stages import SortByColorStage

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

    all_task_attributes = {}
    randomized_config_path = str(Path(__file__).resolve().parent / "detection_randomization.yaml")

    def __init__(self, nb_already_sorted : int = 4) -> None:
        """
        To vary the difficulty, you can change the nb_already_sorted. Easy = 4-5, Medium = 2-3 Hard = 1.
        """
        super().__init__()
        self.env_options = {"nb_cube_completed":nb_already_sorted}

        self.stages = [SortByColorStage(instruction=UserInstruction("Clean the table please"), nb_good_place=nb_already_sorted+1,last=False)]
        for j in range(1, 6-nb_already_sorted):
            self.stages.append(SortByColorStage(instruction=EmptyInstruction(),nb_good_place=nb_already_sorted+j+1, last=(nb_already_sorted+j==5)))
    
        if nb_already_sorted < 2:
            self.approximal_difficulty = "Hard"
        elif nb_already_sorted < 4:
            self.approximal_difficulty = "Medium"
        else:
            self.approximal_difficulty = "Easy"


# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import sapien, torch
from typing import Any, List, Dict
from pathlib import Path

from ..tools.detection_tool import ColorDetectionTools

from magma_core.base.tasks import BaseTask, BaseTaskStage
from magma_core.base.data_structures import UserInstruction, EmptyInstruction, Instruction, Situation
from magma_core.utils.env_utils import is_object_inside_target
from magma_core.base.tasks_style import TaskStyle

class OriginalTaskStage(BaseTaskStage):

    def __init__(self, instruction : Instruction, nb_of_cube : int, rank : int) -> None:
        self.target_steps = 3 if rank == 0 else 2
        self.acceptance_steps = 1 #0 if rank == 0 else 1
        mem = [
                "You are in charge of sorting object by color.",
                "To clean the table, each object must be in the box of the same color"
        ]
        if rank > 0:
            mem.append("The user asked to clean the table.")
            
        self.situation = Situation(
            memory=mem,
            preserved_memory_indices=[0,1],
            attributes={},
            flag_answer_to_user=False,
            instruction=instruction
        )
        self.nb_of_cube = nb_of_cube
        super().__init__(reset_at_end=False, stage_goal_description="The stage goal is to sort a cube")
        

    def verif_env_completion(self, obs: Dict) -> torch.Tensor:

        N = obs["extra"]["agent_tcp"].shape[0]
        device = obs["extra"]["agent_tcp"].device
        nb_yellow = torch.zeros(N, device=device)
        nb_green = torch.zeros(N, device=device)
        err = torch.zeros(N, device=device)

        for obj_name, obj_pose in obs["extra"].items():
            if "green_cube" in obj_name:
                nb_green += is_object_inside_target(obj_pose,obs["extra"]["green_box_pose"],keep_tensor=True).int()
                err += is_object_inside_target(obj_pose,obs["extra"]["yellow_box_pose"],keep_tensor=True).int()
            elif "yellow_cube" in obj_name:
                nb_yellow += is_object_inside_target(obj_pose,obs["extra"]["yellow_box_pose"],keep_tensor=True).int()
                err += is_object_inside_target(obj_pose,obs["extra"]["green_box_pose"],keep_tensor=True).int()

        out = torch.zeros(N, device=device)
        out[((nb_yellow + nb_green) >= self.nb_of_cube)] = 1
        out[(err > 0)] = -1

        return out

class SortColorWithDetection(BaseTask):
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

    def __init__(self, nb_already_sorted : int = 4) -> None:
        """
        To vary the difficulty, you can change the nb_already_sorted. Easy = 4-5, Medium = 2-3 Hard = 1.
        """
        super().__init__()

        self.randomized_config_path = str(Path(__file__).resolve().parent / "detection_randomization.yaml")

        self.env_options = {"nb_cube_completed":nb_already_sorted}

        self.stages = [OriginalTaskStage(instruction=UserInstruction("Clean the table please"), nb_of_cube=nb_already_sorted+1, rank=0)]
        for j in range(1, 6-nb_already_sorted):
            self.stages.append(OriginalTaskStage(instruction=EmptyInstruction(),nb_of_cube=nb_already_sorted+j+1, rank=j))
    
        if nb_already_sorted < 2:
            self.approximal_difficulty = "Hard"
        elif nb_already_sorted < 4:
            self.approximal_difficulty = "Medium"
        else:
            self.approximal_difficulty = "Easy"
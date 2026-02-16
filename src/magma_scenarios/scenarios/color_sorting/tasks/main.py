# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
import sapien, torch
from typing import List, Dict, Literal
from pathlib import Path

from ..tools.simplified_tool import ColorSimplifiedTools

from magma_core.base.tasks import BaseTask, BaseTaskStage
from magma_core.base.data_structures import UserInstruction, Instruction, EmptyInstruction, Situation
from magma_core.utils.env_utils import is_object_inside_target
from magma_core.base.tasks_style import TaskStyle

######## STAGES

attributes = {"objects": ["cube_green_1", "cube_green_2", "cube_green_3",
                "cube_yellow_1", "cube_yellow_2", "cube_yellow_3",
                "green_box", "yellow_box"]}

class StageCubeColor(BaseTaskStage):
    """
    A stage to sort a cube in a box with the same color
    """
    target_steps = 2
    acceptance_steps = 1

    def __init__(self, reset_at_end: bool, instruction : Instruction,
                 stage_goal_description : str, flag_answer_to_user : bool = False,
                 nb_yellow_sorted : int = 0, nb_green_sorted : int = 0) -> None:
        super().__init__(reset_at_end, stage_goal_description=stage_goal_description)
        self.situation = Situation(
            memory=["You are in charge of sorting cubes by color"],
            preserved_memory_indices=[0],
            attributes=attributes,
            instruction=instruction,
            flag_answer_to_user=flag_answer_to_user
        )
        self.nb_yellow_sorted = nb_yellow_sorted
        self.nb_green_sorted = nb_green_sorted

    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        N = obs["extra"]["agent_tcp"].shape[0]
        device = obs["extra"]["agent_tcp"].device
        nb_yellow = torch.zeros(N, device=device)
        nb_green = torch.zeros(N, device=device)
        err = torch.zeros(N, device=device)

        for obj_name, obj_pose in obs["extra"].items():
            if "green_cube" in obj_name:
                nb_green += is_object_inside_target(obj_pose,obs["extra"]["green_box_pose"]).int()
                err += is_object_inside_target(obj_pose,obs["extra"]["yellow_box_pose"]).int()
            elif "yellow_cube" in obj_name:
                nb_yellow += is_object_inside_target(obj_pose,obs["extra"]["yellow_box_pose"]).int()
                err += is_object_inside_target(obj_pose,obs["extra"]["green_box_pose"]).int()

        out = torch.zeros(N, device=device)
        out[(nb_yellow == self.nb_yellow_sorted) & (nb_green == self.nb_green_sorted)] = 1
        out[(err > 0) | (nb_yellow > self.nb_yellow_sorted) | (nb_green > self.nb_green_sorted)] = -1
        return out


class StageCube(StageCubeColor):
    """
    Variant where we are just validating the stage if a cube was added but not with a precise color.
    """

    def __init__(self, total : int, reset_at_end: bool, instruction: Instruction, stage_goal_description : str, flag_answer_to_user: bool = False, nb_yellow_sorted: int = 0, nb_green_sorted: int = 0) -> None:
        super().__init__(reset_at_end, instruction, stage_goal_description, flag_answer_to_user, nb_yellow_sorted, nb_green_sorted)
        self.total = total

    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        N = obs["extra"]["agent_tcp"].shape[0]
        device = obs["extra"]["agent_tcp"].device
        nb_yellow = torch.zeros(N, device=device)
        nb_green = torch.zeros(N, device=device)
        err = torch.zeros(N, device=device)

        for obj_name, obj_pose in obs["extra"].items():
            if "green_cube" in obj_name:
                nb_green += is_object_inside_target(obj_pose,obs["extra"]["green_box_pose"]).int()
                err += is_object_inside_target(obj_pose,obs["extra"]["yellow_box_pose"]).int()
            elif "yellow_cube" in obj_name:
                nb_yellow += is_object_inside_target(obj_pose,obs["extra"]["yellow_box_pose"]).int()
                err += is_object_inside_target(obj_pose,obs["extra"]["green_box_pose"]).int()

        out = torch.zeros(N, device=device)
        out[(nb_yellow + nb_green == self.total)] = 1
        out[(err > 0) | (nb_yellow > self.nb_yellow_sorted) | (nb_green > self.nb_green_sorted)] = -1
        return out



##### TASKS

class OrderedSortColorCube(BaseTask):
    """
    This Tasks consists to asking to the robot to take a specific number of cubes per color and to sort them in their respective box following an precise sequence.
    Difficulty: Easy (1-2 cubes) - Medium (3 cubes) - Hard (4-5-6 cubes)
    """

    name : str = "Sorting Cube by Color Ordered"
    env_id = "SixCubesTwoBoxesOnTable"

    Tools_cls = ColorSimplifiedTools

    styles = [
        TaskStyle.COLOR_REASONING,
        TaskStyle.LONG_STAGE,
        TaskStyle.CONSTRAINED
    ]

    all_task_attributes = attributes

    randomized_config_path = str(Path(__file__).parent.joinpath("detection_randomization.yaml"))

    def __init__(
            self,
            colors : List[Literal["green","yellow"]] = ["green","yellow"], 
            instruction_str : str = "Can you put first a green cube then a yellow one in their respective box?", 
        ) -> None:
        """
        Initiate the task
        
        :param colors: List of cube to sort by color. One color = one cube.
        :type colors: List[Literal["green", "yellow"]]
        :param instruction_str: The instruction to instruct to the robot to sort the object from the list
        :type instruction_str: str
        :param respect_order: If set to True, Object must be sorted exaclty following the order of the mist. But it needs to be specified in the instruction to let the robot knows which one to follows.
        :type respect_order: bool
        """
        super().__init__()
        instruction = UserInstruction(instruction_str)
        self.stages = []
        flag=False
        g=0
        y=0

        total = len(colors)
        if total > 3:
            self.approximal_difficulty = "hard"
        elif total < 3:
            self.approximal_difficulty = "easy"
        else:
            self.approximal_difficulty = "medium"
        
        for i, color in enumerate(colors):
            if color == "green":
                g+=1
                goal = "The objective is to sort a green cube"
            elif color == "yellow":
                y+=1
                goal = "The objective is to sort a yelow cube"
            else:
                raise RuntimeError()
            
            if i == len(colors)-1:
                flag = True

            self.stages.append(StageCubeColor(
                False,
                instruction=instruction,
                flag_answer_to_user=flag,
                stage_goal_description= goal,
                nb_yellow_sorted=y,
                nb_green_sorted=g
                ))
            
            instruction = EmptyInstruction()

class MultipleCubeColorSorting(BaseTask):
    """
    This Tasks consists to asking to the robot to take a specific number of cubes per color and to sort them in their respective box. The model is free of the order.
    Difficulty: Easy (1-2 cubes) - Medium (3 cubes) - Hard (4-5-6 cubes)
    """

    name : str = "Sorting Cube by Color Ordered"
    env_id = "SixCubesTwoBoxesOnTable"

    Tools_cls = ColorSimplifiedTools

    styles = [
        TaskStyle.COLOR_REASONING,
        TaskStyle.LONG_STAGE
    ]

    all_task_attributes = attributes

    randomized_config_path = str(Path(__file__).parent.joinpath("detection_randomization.yaml"))

    def __init__(
            self,
            instruction_str : str = "Can you put 1 green cube and one yellow cube in their respective box please?",
            nb_of_green : int = 1,
            nb_of_yellow : int = 1
        ) -> None:
        """
        Initiate the task
        
        :param colors: List of cube to sort by color. One color = one cube.
        :type colors: List[Literal["green", "yellow"]]
        :param instruction_str: The instruction to instruct to the robot to sort the object from the list
        :type instruction_str: str
        :param respect_order: If set to True, Object must be sorted exaclty following the order of the mist. But it needs to be specified in the instruction to let the robot knows which one to follows.
        :type respect_order: bool
        """
        super().__init__()
        instruction = UserInstruction(instruction_str)
        self.stages = []
        flag=False

        total = nb_of_green+nb_of_yellow
        if total > 4:
            self.approximal_difficulty = "hard"
        elif total < 3:
            self.approximal_difficulty = "easy"
        else:
            self.approximal_difficulty = "medium"
        
        for i in range(total):
            if i == total-1:
                flag = True

            self.stages.append(StageCube(
                reset_at_end=False,
                instruction=instruction,
                flag_answer_to_user=flag,
                stage_goal_description="The stage goal is just to sort one cube (color does not matter)",
                nb_yellow_sorted=nb_of_yellow,
                nb_green_sorted=nb_of_green,
                total=i+1,
            ))
            
            instruction = EmptyInstruction()


class SeqSortColorCube(BaseTask):
    """
    Consist to ask to the model multiple time in a row to sort one cube. 
    The last stage consist to ask the model to equalize the number of cube per box by adding new ones.

    Difficulty range : Medium to Hard
    """

    name : str = "Sequenced Sorting by Color"
    env_id = "SixCubesTwoBoxesOnTable"

    Tools_cls = ColorSimplifiedTools

    styles = [
        TaskStyle.COLOR_REASONING
    ]
    all_task_attributes = attributes
    randomized_config_path = str(Path(__file__).parent.joinpath("detection_randomization.yaml"))

    def __init__(self, hard_mode : bool = False) -> None:
        """
        If Hard mode is set it add some additional steps.
        """
        super().__init__()


        self.stages = [
            StageCubeColor(
                reset_at_end=False,
                instruction=UserInstruction("Clean one green cube please"),
                stage_goal_description= "The stage objective is to sort a green cube",
                flag_answer_to_user=True,
                nb_green_sorted=1,
            ),
        ]

        if hard_mode:
            self.approximal_difficulty = "hard"
            self.stages.extend(
                [StageCubeColor(
                reset_at_end=False,
                instruction=UserInstruction("Put two yellow in their box now."),
                stage_goal_description= "The stage objective is to sort a yellow cube",
                flag_answer_to_user=False,
                nb_green_sorted=1,
                nb_yellow_sorted=1,
                ),
                StageCubeColor(
                    reset_at_end=False,
                    instruction=EmptyInstruction(),
                    flag_answer_to_user=True,
                    stage_goal_description= "The stage objective is to sort a yellow cube",
                    nb_green_sorted=1,
                    nb_yellow_sorted=2,
                )]
            )
            y=2
        else:
            self.approximal_difficulty = "medium"
            self.stages.append(
                StageCubeColor(
                    reset_at_end=False,
                    instruction=UserInstruction("Put one yellow cube in their box now."),
                    flag_answer_to_user=True,
                    stage_goal_description= "The stage objective is to sort a yellow cube",
                    nb_green_sorted=1,
                    nb_yellow_sorted=1,
                )
            )
            y=1
        
        self.stages.extend([
            StageCubeColor(
                reset_at_end=False,
                instruction=UserInstruction("Okay one more yellow cube please."),
                flag_answer_to_user=True,
                stage_goal_description= "The stage objective is to sort a yellow cube",
                nb_green_sorted=1,
                nb_yellow_sorted=y+1,
            ),
            StageCubeColor(
                reset_at_end=False,
                instruction=UserInstruction("Perfect, Great job! Now I want that there is in the green box the same amount of green cube that yellow cube in the yellow box"),
                flag_answer_to_user=False if hard_mode else True,
                stage_goal_description= "The stage objective is to sort a green cube",
                nb_green_sorted=2,
                nb_yellow_sorted=y+1,
            )
        ])

        if hard_mode:
            self.stages.append(
                StageCubeColor(
                    reset_at_end=False,
                    instruction=EmptyInstruction(),
                    flag_answer_to_user=True,
                    stage_goal_description= "The stage objective is to sort a green cube",
                    nb_green_sorted=3,
                    nb_yellow_sorted=3,
                )
            )
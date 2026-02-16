# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.data_structures import Situation, Instruction, UserInstruction
from magma_core.base.tasks import BaseTask, BaseTaskStage
from magma_core.utils.env_utils import is_object_inside_target

from ..tool import BiRobotTools
from magma_core.base.tasks_style import TaskStyle

import torch
from typing import Any, List, Dict, Optional
from importlib.resources import files

ROBOT_NAMES = ["arm1","arm2"]

class BiRobotSituation(Situation):

    def __init__(
            self, 
            memory: List[str], 
            instruction: Instruction, 
            user_scenario: str = "none",
            history = None) -> None:
        super().__init__(
            memory,
            [0,1],
            {"known_area": ["right", "left", "mutual"], "known_robots": ROBOT_NAMES},
            instruction,
            flag_answer_to_user=True,
            user_scenario=user_scenario,
            history=history
            )

class SimpleTransfertOfObject(BaseTaskStage):

    target_steps = 1
    acceptance_steps = 2

    
    stage_goal_description = "The goal is to send the object to right zone (initially it's mutual)"

    situation = BiRobotSituation(
        memory=[
            "Robot 1 has access to left area and Robot 2 has access to right area",
            "Both robots have access to mutual area.",
            "The user asked about the position of the obj_A"
        ],
        instruction=UserInstruction("Okay I need it to the right zone please"),
        user_scenario="""The robot is asked to move an object to the right zone, however it does not know how the object is named.
        Therefore it needs to call a function to check the status of each objects before being able to deplace the object.

        If the robot tells you about errors to find an object you must tell it to check the object name using the adapted function.
        If the robot tells you that the object is out of range for the robot 1, tell him to use the robot 2 instead.
        If you do not know what to answer or you hesitate just answer 'STOP'.
        """
    )

    def __init__(self, reset_at_end: bool) -> None:
        super().__init__(reset_at_end, self.stage_goal_description)
        
    
    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        return is_object_inside_target(obs["extra"]["obj_A"], obs["extra"]["right_zone"], keep_tensor = True).int()
    
class AskForObjectPose(BaseTaskStage):

    target_steps = 2
    acceptance_steps = 2

    stage_goal_description = "The goal here is to call the object state function to find the obj_A"

    situation = BiRobotSituation(
        memory=[
            "Robot 1 has access to left area and Robot 2 has access to right area",
            "Both robots have access to mutual area."
        ],
        instruction=UserInstruction("Hello, Where is the obj_A?"),
        user_scenario="""The robot is asked to tell you the position of the obj_A.

        If the robot tells you that it do not know any obj_A, asks him to use its detection function to get objects state.
        If the robot tells you something else, just ask him the position of the obj_A. You can be a bit upset on the answer.
        """
    )
        
    verification_prompt = "The robot must simply answer that the obj_A is in the mutual zone"

    def __init__(self, reset_at_end: bool) -> None:
        super().__init__(reset_at_end, self.stage_goal_description)

    
class ReverseTransfertOfObject(BaseTaskStage):

    target_steps = 2
    acceptance_steps = 1

    stage_goal_description = "The goal is to send the obj_A to left_zone (Initially it's right so it's uspposed to stop first in mutual then left after)"

    situation = BiRobotSituation(
        memory=[
            "Robot 1 has access to left area and Robot 2 has access to right area",
            "Both robots have access to mutual area.",
            "I deposed the object to right zone as user asked."
        ],
        instruction=UserInstruction("Ha, I made an error. I need it to left zone instead"),
    )
    
    def __init__(self, reset_at_end: bool) -> None:
        super().__init__(reset_at_end, self.stage_goal_description)
    
    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        return is_object_inside_target(obs["extra"]["obj_A"], obs["extra"]["left_zone"], keep_tensor = True).int()


class BiRobotSortSimple(BaseTask):
    """
    The idea is simply to transport an object from a zone to another using two robots. Each robot have access to a specific zone + mutual one.
    The model must understand that it needs to use different robots in a row to successfully brings the objects to the correct zone.
    Difficulty is Medium.
    """

    name : str = "BI Robot Sorting Cube"

    # randomized_config_path = str(files(__package__).joinpath("tools_with_color.yaml"))

    env_id = "BiRobotSorting-v1"

    styles = [
        TaskStyle.MULTI_ROBOT
    ]
    
    Tools_cls = BiRobotTools

    approximal_difficulty = "Medium"

    agent_names = ROBOT_NAMES

    def __init__(self) -> None:
        super().__init__()
        self.stages = [
            AskForObjectPose(reset_at_end=True),
            SimpleTransfertOfObject(reset_at_end=False),
            ReverseTransfertOfObject(reset_at_end=False)
        ]
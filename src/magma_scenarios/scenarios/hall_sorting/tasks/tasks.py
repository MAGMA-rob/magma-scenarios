# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.data_structures import Situation, Instruction, UserInstruction
from magma_core.base.tasks import BaseTask, BaseTaskStage
from magma_core.utils.env_utils import is_object_inside_target

from ..tools.simplified import HallTool
from magma_core.base.tasks_style import TaskStyle

import torch
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass

ROBOT_NAMES = ["robot1","robot2"]

@dataclass
class HallAssignment:
    hall_name : str
    objects : List[Tuple[str, int]]

    def verif(self, obs : Dict) -> torch.Tensor:
        device = obs["extra"][self.hall_name].device
        hall_pose = obs["extra"][self.hall_name]
        batch_size = hall_pose.shape[0]

        K = len(self.objects)
        counts = torch.zeros(batch_size, K, device=device, dtype=torch.int32)

        for k, (obj_key, nb) in enumerate(self.objects):
            for name, obj_pose in obs["extra"].items():
                if obj_key in name:
                    inside = is_object_inside_target(
                        obj_pose, hall_pose, thresh=0.2
                    ).int()                      # (B,)
                    counts[:, k] += inside

        target = torch.tensor(
            [nb for _, nb in self.objects],
            device=device,
            dtype=torch.int32
        )

        return (counts == target).all(dim=1)  # (B,) bool
                        

class HallSortingBaseSituation(Situation):

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

    
class ObjectInHall(BaseTaskStage):

    target_steps = 2
    acceptance_steps = 1

    stage_goal_description = "The goal is to send the obj_A to left_zone (Initially it's right so it's uspposed to stop first in mutual then left after)"

    # situation = Situation(
    #     memory=[
    #         "Robot 1 has access to left area and Robot 2 has access to right area",
    #         "Both robots have access to mutual area.",
    #         "I deposed the object to right zone as user asked."
    #     ],
    #     instruction=UserInstruction("Ha, I made an error. I need it to left zone instead"),
    # )
    
    def __init__(self, reset_at_end: bool, hall_assignment : List[HallAssignment]) -> None:
        ...
    
    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        return is_object_inside_target(obs["extra"]["obj_A"], obs["extra"]["left_zone"], keep_tensor = True).int()


class BaseHallSortingTask(BaseTask):
    """
    The idea is simply to transport an object from a zone to another using two robots. Each robot have access to a specific zone + mutual one.
    The model must understand that it needs to use different robots in a row to successfully brings the objects to the correct zone.
    Difficulty is Medium.
    """

    name : str = "Hall Sorting"

    # randomized_config_path = str(files(__package__).joinpath("tools_with_color.yaml"))

    env_id = "4HallSorting"

    styles = [
        TaskStyle.MULTI_ROBOT,
        TaskStyle.DETECTION_TASK,
    ]
    
    Tools_cls = HallTool

    approximal_difficulty = "Medium"

    agent_names = ROBOT_NAMES
    all_task_attributes = {
        "known_robots": ROBOT_NAMES,
        "robot1_bag": [],
        "robot2_bag": [],
        "halls" : ["Hall1","Hall2","Hall3","Hall4"]
    }

    def __init__(self) -> None:
        super().__init__()
        self.stages = []
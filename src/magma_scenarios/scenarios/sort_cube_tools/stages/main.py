# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tasks import BaseTaskStage, ConstraintBaseStage
from magma_core.base.data_structures import UserInstruction, EmptyInstruction, Situation
from magma_core.utils.env_utils import is_object_inside_target

import sapien, torch
from typing import List, Dict, Optional

from importlib import resources
import yaml

att = {"object_name":["red_cube", "blue_cube", "green_cube", "white_box"]}

class Constraint(ConstraintBaseStage):
    
    def __init__(self, constraint: str, memory: List[str]) -> None:
        super().__init__(constraint, memory, att)

class CubeInBox(BaseTaskStage):
    """
    Stage to put a cube inside the box
    """

    target_steps = 2
    acceptance_steps = 1

    def __init__(self, reset_at_end: bool, obj_name : str, mem : List[str], instruction : Optional[str]) -> None:
        super().__init__(reset_at_end, f"The goal of the stage is to put {obj_name} in the box")
        self.obj_name = obj_name
        if not instruction:
            ins = EmptyInstruction()
        else:
            ins = UserInstruction(instruction)
        
        self.situation = Situation(
            memory=mem,
            attributes=att,
            instruction=ins,
            flag_answer_to_user=False,
            preserved_memory_indices=[]
        )   
    

    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        return is_object_inside_target(obs["extra"][self.obj_name], obs["extra"]["white_box"])

class CubeOnAnother(BaseTaskStage):

    target_steps = 2
    acceptance_steps = 1

    def __init__(self, reset_at_end: bool, top_object : str, bot_object : str,
                 instruction : Optional[str], memory : List[str]) -> None:
        super().__init__(reset_at_end, f"The goal of this stage is to put the {top_object} on the {bot_object}")
        self.top = top_object
        self.bot = bot_object

        if not instruction:
            ins = EmptyInstruction()
        else:
            ins = UserInstruction(instruction)

        self.situation = Situation(
            memory=memory,
            attributes=att,
            instruction=ins,
            flag_answer_to_user=False,
            preserved_memory_indices=[]
        )

    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        stacked = is_object_inside_target(obs["extra"][self.top],obs["extra"][self.bot], thresh=0.02)
        on_top = obs["extra"][self.top][: , 2] > obs["extra"][self.bot][: , 2]
        out = stacked.int() & on_top
        return out
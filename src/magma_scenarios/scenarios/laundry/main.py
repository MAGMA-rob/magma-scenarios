# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import random
from typing import List, cast, Any, Dict
from pathlib import Path

import torch

from magma_core.utils.env_utils import is_object_inside_target

from .load_tool import LaunchTool
from magma_core.base.tasks import BaseTask, BaseTaskStage
from magma_core.base.tasks_style import TaskStyle
from magma_core.base.data_structures import Situation, EmptyInstruction, UserInstruction, Instruction, Log

all_clothes = [
    "shirt",
    "blouse",
    "jeans",
    "pants",
    "short",
    "socks",
    "boxer",
    "panties",
    "cap",
]

class Verification:

    to_clean : List[str]

    def __init__(self, nb_of_element_to_wash : int) -> None:
        if nb_of_element_to_wash > len(all_clothes):
            raise ValueError(f"The passed NB ({nb_of_element_to_wash}) is > to the maximal length of available clothes ({len(all_clothes)})")
        self.to_clean = random.sample(all_clothes,nb_of_element_to_wash)

    def build_instruction(self) -> Instruction:
        return UserInstruction("Can you clean " + " and ".join(self.to_clean))


class LoadClotheStage(BaseTaskStage):
    """
    Stage to load desired clothes in the machine
    """

    target_steps = 2
    acceptance_steps = 1

    def __init__(self, n : int, clothes_to_load : List[str], instruction : Instruction = EmptyInstruction()) -> None:
        super().__init__(False, f"The goal of this stage is to have {n} clothes from the list {clothes_to_load} in the washing machine")
        self.clothes = clothes_to_load
        self.n = n
        self.situation = Situation(
            memory = [
                "To wash clothes, I need to put them inside the wash-machine, add detergents and then use 'wash'.",
                "Detergent must always be put last in the wash-machine"
            ],
            preserved_memory_indices= [0],
            attributes={
                "clothes": all_clothes.copy(),
                "additionals": ["detergent"],
            },
            instruction = instruction,
            flag_answer_to_user=False
        )

    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        cpt = torch.zeros(
            obs["extra"]["washing_machine"]["pose"].shape[0],
            device=obs["extra"]["washing_machine"]["pose"].device,
            dtype=torch.int32,
        )

        for clothe in self.clothes:
            cpt += is_object_inside_target(
                obs["extra"][clothe]["pose"],
                obs["extra"]["washing_machine"]["pose"],
                keep_tensor=True,
            ).int()

        out = (cpt >= self.n).int()

        return out


class WashStage(BaseTaskStage):
    """
    Stage for putting detergent and then launching the washing machine.
    """

    target_steps = 3
    acceptance_steps = 1

    def __init__(self, clothes : List[str]) -> None:
        super().__init__(True, "The goal of this stage is to finally start the washing machine with the detergent inside")
        self.to_clean = clothes
        self.situation = Situation(
            memory = [
                "To wash clothes, I need to put them inside the wash-machine, add detergents and then use 'wash'.",
                "Detergent must always be put last in the wash-machine"
            ],
            preserved_memory_indices= [0],
            attributes={
                "clothes": all_clothes.copy(),
                "additionals": ["detergent"],
            },
            instruction= EmptyInstruction(),
            flag_answer_to_user=True
        )

    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        return is_object_inside_target(obs["extra"]["detergent"]["pose"],obs["extra"]["washing_machine"]["pose"])
    
    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if len(stage_log) == 0:
            return 0
        
        for clothe in self.to_clean:
            if not clothe in stage_log[-1].content:
                return -1
            
        return 1

class Laundry(BaseTask):
    """
    The scene contains clothes and others objects. The aim is to put all dirty clothes into the washing machine, add soap and start a cycle.

    DIfficulty range from medium to hard.
    """

    name = "Laundry"
    env_id = "Laundry-v1"

    randomized_config_path = str(Path(__file__).parent.joinpath("laundry.yaml"))

    Tools_cls = LaunchTool

    styles = [
        TaskStyle.LONG_STAGE
    ]

    def __init__(self, number_of_clothes : int = 1):
        """
        You can modify the number_of_clothes to augment the number of clothes to put in the machine.
         < 3 is Medium but > 3 and < 8 is Hard.
        """
        super().__init__()
        self.verif_elem = Verification(number_of_clothes)

        instruction = self.verif_elem.build_instruction()
        print(instruction.get_content())

        self.stages = [LoadClotheStage(1,self.verif_elem.to_clean,instruction)]
        for i in range(2, number_of_clothes+1):
            self.stages.append(LoadClotheStage(i,self.verif_elem.to_clean))
        self.stages.append(WashStage(self.verif_elem.to_clean))

        self.approximal_difficulty = "Medium" if number_of_clothes < 3 else "Hard"
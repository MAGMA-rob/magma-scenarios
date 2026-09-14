# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
from pathlib import Path
from typing import List
import random

from .load_tool import LaunchTool
from .laundry_stages import WashStage, LoadClotheStage
from .attributes import all_clothes, all_detergents

from magma_core.simulation.tasks import BaseTask
from magma_core.simulation.stage import ConstraintBaseStage
from magma_core.simulation.data_structures import (
    Instruction, UserInstruction, SituationInit
)

RANDOMIZED_CONFIG_PATH = str(Path(__file__).resolve().parent / "laundry.yaml")

class BaseLaundry(BaseTask):

    maniskill_env_id = "Laundry-v1"
    Tools_cls = LaunchTool

    situation_init = SituationInit(
        attributes={
            "clothes" : all_clothes,
            "detergents": all_detergents
        },
        memory={"memory_list":[
            "To wash clothes, I need to put them inside the wash-machine, add detergents and then use 'wash'.",
            "Detergent must always be put last in the wash-machine",
            "Different detergents must not mixed in the same wash."
        ]}
    )

    randomized_config_path = RANDOMIZED_CONFIG_PATH

    def __init__(self, number_of_clothes : int = 2, total : int = 6) -> None :
        """
        Total is the total number of clothes that we want to use for the task
        number_of_clothes is the number of clothes that we want to ask to clean
        """
        super().__init__()

        self.all_detergents = all_detergents.copy()
        self.clothes = all_clothes.copy()
        random.shuffle(self.clothes)
        self.clothes = self.clothes[:total]

        self.verif_elem = Verification(number_of_clothes, self.clothes)
        self.instruction = self.verif_elem.build_instruction()

        self.cloth_to_detergent = {cloth : random.choice(self.all_detergents) for cloth in self.clothes}


class Verification:

    to_clean : List[str]

    def __init__(self, nb_of_element_to_wash : int, clothes : List[str]) -> None:
        if nb_of_element_to_wash > len(clothes):
            raise ValueError(f"The passed NB ({nb_of_element_to_wash}) is > to the maximal length of available clothes ({len(all_clothes)})")
        self.to_clean = random.sample(clothes,nb_of_element_to_wash)

    def build_instruction(self) -> Instruction:
        return UserInstruction("Can you clean " + " and ".join(self.to_clean))
    


class LaundryFromDetergentPreset(BaseLaundry):
    """
    input = detergent
    output = clothes à laver
    """
    name = "Laundry from detergent"

    def __init__(self,number_of_clothes : int = 3) :
        super().__init__(number_of_clothes)

        assigned_detergents = sorted(set(self.cloth_to_detergent.values()))
        self.target_detergent = random.choice(assigned_detergents)
        self.target_clothes = [cloth for cloth, detergent in self.cloth_to_detergent.items() if detergent == self.target_detergent]

        desc = ", ".join(f"{cloth} uses {detergent}" for cloth, detergent in self.cloth_to_detergent.items())
        instruction = UserInstruction(f"Wash all clothes that can be washed with {self.target_detergent}")
        self.stages = [ConstraintBaseStage(desc)]
        self.stages.append(LoadClotheStage(1,self.target_clothes,instruction))
        for i in range(2, len(self.target_clothes)+1):
            self.stages.append(LoadClotheStage(i,self.target_clothes))
        self.stages.append(WashStage(self.target_detergent, self.target_clothes))

class LaundryCompatibleClothesPreset(BaseLaundry):
    """
    input = clothes
    condition = même detergent
    """
    name = "Laundry compatible clothes"

    def __init__(self,number_of_clothes : int = 3) :
        super().__init__(number_of_clothes)

        assigned_detergents = sorted(set(self.cloth_to_detergent.values()))
        self.target_detergent = random.choice(assigned_detergents)
        self.target_clothes = [cloth for cloth, detergent in self.cloth_to_detergent.items() if detergent == self.target_detergent]
        
        desc = ",".join(f"{cloth} uses {detergent} detergent" for cloth, detergent in self.cloth_to_detergent.items())
        instruction = UserInstruction("Please wash the following clothes: " + ", ".join(f"{cloth}" for cloth in self.target_clothes ))
        self.stages = [ConstraintBaseStage(desc)]
        
        self.stages.append(LoadClotheStage(1,self.target_clothes,instruction))
        for i in range(2, len(self.target_clothes)+1):
            self.stages.append(LoadClotheStage(i,self.target_clothes))
        self.stages.append(WashStage(self.target_detergent, self.target_clothes))

        self.approximal_difficulty = "Medium" if number_of_clothes < 3 else "Hard"

    

# class LaundryIncompatibleClothesPreset(BaseLaundry):
#     """
#     input = clothes
#     condition = detergents différents → refuse
#     """
#     def __init__(self,number_of_clothes : int = 3) :
#         super().__init__(number_of_clothes)

#         self.target_detergent = random.choice(self.all_detergents)
#         self.target_clothes = [cloth for cloth, detergent in self.cloth_to_detergent.items() if detergent == self.target_detergent]
#         self.intrusion_clothes = [cloth for cloth,detergent in self.cloth_to_detergent.items() if detergent != self.target_detergent]
#         self.intrusion_cloth = random.choice(self.intrusion_clothes)
#         self.target_clothes.append(self.intrusion_cloth)

#         desc = ','.join(f"{cloth} use {detergent} dtergent" for cloth, detergent in self.cloth_to_detergent.items())
        
#         instruction_which_must_fail = "please wash the following clothes :" + ",".join(f"{cloth}" for cloth in self.target_clothes)
#         instruction_which_must_succeed = UserInstruction("")

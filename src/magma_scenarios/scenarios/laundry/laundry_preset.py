# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
from pathlib import Path
from typing import List
import random

from .load_tool import LaunchTool
from .laundry_stages import WashStage, LoadClotheStage, ContraintWashStage
from .attributes import all_clothes, all_detergent

from magma_core.base.tasks import BaseTask
from magma_core.base.tasks_style import TaskStyle
from magma_core.base.data_structures import Instruction, UserInstruction

class Verification:

    to_clean : List[str]

    def __init__(self, nb_of_element_to_wash : int) -> None:
        if nb_of_element_to_wash > len(all_clothes):
            raise ValueError(f"The passed NB ({nb_of_element_to_wash}) is > to the maximal length of available clothes ({len(all_clothes)})")
        self.to_clean = random.sample(all_clothes,nb_of_element_to_wash)

    def build_instruction(self) -> Instruction:
        return UserInstruction("Can you clean " + " and ".join(self.to_clean))

class SimplePreset(BaseTask):
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

        self.stages = [LoadClotheStage(1,self.verif_elem.to_clean,instruction)]
        for i in range(2, number_of_clothes+1):
            self.stages.append(LoadClotheStage(i,self.verif_elem.to_clean))
        self.stages.append(WashStage(self.verif_elem.to_clean))

        self.approximal_difficulty = "Medium" if number_of_clothes < 3 else "Hard"

class TeamLaundryPreset(BaseTask):
    """
    load clohes
    select correct detergent by clothes
    take detergent
    wash machine
    """

    env_id = "Laundry-v1"
    Tools_cls = LaunchTool
    name = "laundry assignemnt"
    styles = []
    approximal_difficulty = "Hard"
    def __init__(self,number_of_clothes : int = 3):
        
        super().__init__()

        self.verif_elem = Verification(number_of_clothes)

        instruction = self.verif_elem.build_instruction()

        self.detergents = all_detergent.copy()
        self.clothes = all_clothes.copy()

        cloth_to_detergent = {cloth : random.choice(self.detergents) for cloth in self.verif_elem.to_clean}

        self.tools_constant["cloth_to_detergent"] = cloth_to_detergent
        #difference between 'contraint' et 'mem' specialy in the exemple of coffe ConstrainedPreset
        #how to test the task/preset
        self.stages = [ContraintWashStage("to wash the clothes you must select the correct detergent and put it inside the wash-machine before using 'wash'.",cloth_to_detergent)]
        self.stages.append(LoadClotheStage(1,self.verif_elem.to_clean,instruction))
        for i in range(2, number_of_clothes+1):
            self.stages.append(LoadClotheStage(i,self.verif_elem.to_clean))
        self.stages.append(WashStage(cloth_to_detergent))


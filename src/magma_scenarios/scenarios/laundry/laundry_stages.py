import random
from typing import List, Dict
import torch

from magma_core.utils.env_utils import is_object_inside_target
from magma_core.base.stage import BaseTaskStage
from magma_core.base.data_structures import Situation, EmptyInstruction, UserInstruction, Instruction, Log
from magma_core.base.goals import CountAt, At

from .attributes import all_clothes

class LoadClotheStage(BaseTaskStage):
    """
    Stage to load desired clothes in the machine
    """

    target_steps = 2
    acceptance_steps = 1

    def __init__(self, n : int, clothes_to_load : List[str], instruction : Instruction = EmptyInstruction()) -> None:       
        super().__init__([CountAt(clothes_to_load,"washing_machine_basket",n)],False, f"The goal of this stage is to have {n} clothes from the list {clothes_to_load} in the washing machine")
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

class WashStage(BaseTaskStage):
    """
    Stage for putting detergent and then launching the washing machine.
    """

    target_steps = 3
    acceptance_steps = 1

    def __init__(self, clothes : List[str]) -> None:
        super().__init__([At("detergent","washing_machine_basket")],True, "The goal of this stage is to finally start the washing machine with the detergent inside")
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

    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if len(stage_log) == 0:
            return 0
        
        if len(stage_log[-1].content) != len(self.to_clean):
            return -1

        for clothe in self.to_clean:
            if not clothe in stage_log[-1].content:
                return -1
        
        return 1
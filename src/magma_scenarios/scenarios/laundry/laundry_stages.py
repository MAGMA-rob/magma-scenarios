import random
from typing import List, Dict
import torch

from magma_core.utils.env_utils import is_object_inside_target
from magma_core.base.stage import BaseTaskStage, ConstraintBaseStage
from magma_core.base.data_structures import Situation, EmptyInstruction, UserInstruction, Instruction, Log
from magma_core.base.goals import At, AtLeastCountAt

from .attributes import all_clothes, all_detergents
from .laundry_errors import GraspClothesFailureError

class LoadClotheStage(BaseTaskStage):
    """
    Stage to load desired clothes in the machine
    """

    target_steps = 2
    acceptance_steps = 1

    def __init__(self, n : int, clothes_to_load : List[str], instruction : Instruction = EmptyInstruction()) -> None:       
        super().__init__([AtLeastCountAt(clothes_to_load,"washing_machine_basket",n)],False, f"The goal of this stage is to have {n} clothes from the list {clothes_to_load} in the washing machine")
        self.possible_errors = [
            GraspClothesFailureError(clothes_to_load)
        ]
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

    def __init__(self, target_detergent : str , target_clothes : list ) -> None:
        super().__init__([],
        True, 
        "The goal of this stage is to wash all clothes using the correct detergent for each item")

        self.to_clean = target_clothes
        self.target_detergent = target_detergent
        self.situation = Situation(
            memory = [],
            preserved_memory_indices= [],
            attributes={
                "clothes": all_clothes.copy(),
                "detergents": all_detergents.copy(),
            },
            instruction= EmptyInstruction(),
            flag_answer_to_user=True
        )

    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if len(stage_log) == 0:
            return 0
        
        if len(stage_log[-1].content["clothes" ]) != len(self.to_clean):
            return -1

        for clothe in self.to_clean:
            if not clothe in stage_log[-1].content["clothes"]:
                return -1

        if len(stage_log[-1].content["detergent"]) > 1 :
            return -1
        
        if self.target_detergent not in stage_log[-1].content["detergent"] :
            return -1
        
        return 1

class ContraintWashStage(ConstraintBaseStage):
    def __init__(self,contraint : str) -> None:

        mem = ["you must not mix different detergents in the same wash.",
                "each cloth requires a specific detergent."]
        
        super().__init__(contraint,mem,{
                "clothes": all_clothes.copy(),
                "detergents": all_detergents.copy(),
            },)



class RefuseLaundryStage(BaseTaskStage):
    acceptance_steps = 0
    target_steps = 1

    def __init__(self, instruction, verif_prompt : str) -> None:
        super().__init__([], False, "")

        self.situation = Situation(
            memory=[],
            preserved_memory_indices=[],
            attributes={
                "all_clothes" : all_clothes.copy(),
                "all_detergents" : all_detergents.copy()
            },
            flag_answer_to_user=False,
            instruction=UserInstruction(instruction),
        )

        self.verification_prompt = verif_prompt
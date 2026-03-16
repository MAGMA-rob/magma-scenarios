# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.stage import BaseTaskStage, ConstraintBaseStage, ModifAttributesBaseStage
from magma_core.base.data_structures import UserInstruction, EmptyInstruction, Log, Situation
from magma_core.base.goals import At, NotAt
from magma_core.base.data_structures.situation import Instruction

from typing import List, Dict, Optional
from .att import AREAS

class ObjectToZone(BaseTaskStage):

    target_steps = 2
    acceptance_steps = 1

    def __init__(
            self,
            assignement : Dict[str,str],
            all_objects : List[str],
            instruction : str = "none"
        ) -> None:
        """
        Docstring for __init__
        
        :param instruction: The instruction to give to the user. If not specified, it will create an empty instruction (supposing that you gave previosuly an instruction to launch multiple cycle)
        :param instruction: str.
        :param assignement: Dict of assignment. Objects as key and area as value. Objects which should not be manipulated must have 'none' as value.
        :type assignement: Dict[str, str]
        :param all_area: The list of all existing area.
        :type all_area: List[str]
        """
        goals = []
        for obj in all_objects:
            target = assignement.get(obj, None)
            forbidden_area = []
            for area in AREAS:
                if area == target:
                    goals.append(At(obj,area))
                else:
                    forbidden_area.append(area)
            if forbidden_area:
                goals.append(NotAt(obj,forbidden_area,True))


        super().__init__(
            goals,
            reset_at_end=True,
            stage_goal_description=f"The goal of this stage is to sort only one object per class without using the cycle tool (use only take and drop) according to {assignement}"
        )

        if instruction == "none":
            query = EmptyInstruction()
        else:
            query = UserInstruction(instruction)

        self.situation = Situation(
            memory=["You are in charge of sorting objects in a factory."],
            preserved_memory_indices=[0],
            instruction=query,
            attributes={
                "objects" : all_objects,
                "target_areas" : AREAS,
            },
            flag_answer_to_user=False
        )

    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        for l in stage_log:
            if l.function == "launch_cycle":
                return -1
        return 1
        
class ConstraintSorting(ConstraintBaseStage):

    def __init__(self, constraint: str, add_memory: List[str], attributes: Dict) -> None:
        memory = [
            "You are in charge of sorting objects in a factory."
        ]
        memory.extend(add_memory)
        super().__init__(constraint, memory, attributes)


class AddLocationStage(ModifAttributesBaseStage):

    def __init__(self, instruction: Instruction, val_name: str, memory: List[str], attributes: Dict, flag_answer_to_user: bool = True) -> None:
        super().__init__("ADD", instruction, val_name, "target_areas", memory, [], attributes, flag_answer_to_user)
    
class RemoveLocationStage(ModifAttributesBaseStage):

    def __init__(self, instruction: Instruction, val_name: str, memory: List[str], attributes: Dict, flag_answer_to_user: bool = True) -> None:
        super().__init__("REMOVE", instruction, val_name, "target_areas", memory, [], attributes, flag_answer_to_user)
    
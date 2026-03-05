# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tasks import BaseTaskStage, ConstraintBaseStage, ModifAttributesBaseStage
from magma_core.base.data_structures import UserInstruction, EmptyInstruction, Log, Situation
from magma_core.base.data_structures.situation import Instruction
from magma_core.utils.env_utils import is_object_inside_target

import sapien, torch
from typing import List, Dict, Literal, Tuple, Optional, Any
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
        super().__init__(reset_at_end=True, stage_goal_description=f"The goal of this stage is to sort only one object per class without using the cycle tool (use only take and drop) according to {assignement}")
        self.assignment = assignement
        self.obj = all_objects

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

    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        nb_envs = obs["extra"]["area1"].shape[0]
        device_1 = obs["extra"]["area1"].device
        out = torch.zeros(nb_envs, dtype=torch.int8, device=device_1)

        verify = torch.ones(nb_envs, dtype=torch.bool, device=device_1)
        any_error = torch.zeros(nb_envs, dtype=torch.bool, device=device_1)

        for obj_name in self.obj:
            pos = obs["extra"][obj_name]
            target = self.assignment.get(obj_name, None)
            if target is not None:
                verify &= is_object_inside_target(pos, obs["extra"][target])

            for area in AREAS:
                if target is not None and area != target:
                    any_error |= is_object_inside_target(pos, obs["extra"][area])

        out[any_error] = -1
        out[~any_error & verify] = 1

        return out.int()

    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        for l in stage_log:
            if l.function == "launch_cycle":
                return -1
        return 1

class Cycle(BaseTaskStage):

    target_steps = 1
    acceptance_steps = 1

    def __init__(
            self,
            assignement : Dict[str,str],
            known_areas : List[str],
            flag_answer : bool,
            manu_order : Optional[str] = None,
            instruction : str = "none",
        ) -> None:
        """
        Docstring for __init__
        
        :param instruction: The instruction to give to the user. If not specified, it will create an empty instruction (supposing that you gave previosuly an instruction to launch multiple cycle)
        :param instruction: str.
        :param assignement: Dict of assignment. Objects as key and area as value. Objects which should not be manipulated must have 'none' as value.
        :type assignement: Dict[str, str]
        :param all_area: The list of all existing area.
        :type all_area: List[str]
        :param manu_order: The optional desired manufacturing order.
        :type manu_order: Optional[str]
        """
        super().__init__(reset_at_end=True, stage_goal_description=f"The goal of this stage is to sort all objects according to the assignment provided by the user and the memory : {assignement}")
        self.assignment = assignement
        self.manu_order = manu_order
        self.areas = known_areas

        if instruction == "none":
            query = EmptyInstruction()
        else:
            query = UserInstruction(instruction)

        self.situation = Situation(
            memory=["You are in charge of sorting objects in a factory."],
            preserved_memory_indices=[0],
            instruction=query,
            attributes={
                "objects" : list(assignement.keys()),
                "target_areas" : known_areas,
            },
            flag_answer_to_user=flag_answer
        )


    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        nb_envs = obs["extra"]["area1"].shape[0]
        device_1 = obs["extra"]["area1"].device
        out = torch.zeros(nb_envs, dtype=torch.int8, device=device_1)

        verify = torch.ones(nb_envs, dtype=torch.bool, device=device_1)
        any_error = torch.zeros(nb_envs, dtype=torch.bool, device=device_1)
        for obj_name, associated_area in self.assignment.items():
            pos = obs["extra"][obj_name]
            if associated_area != "none":
                verify &= is_object_inside_target(pos, obs["extra"][associated_area])
            for area in self.areas:
                if area != associated_area:
                    any_error |= is_object_inside_target(pos, obs["extra"][area])

        out[any_error] = -1
        out[~any_error & verify] = 1
        return out.int()

    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if self.manu_order is None: return 1
        if len(stage_log) == 0: return 0
        if stage_log[-1].content == self.manu_order: return 1
        return -1
        
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
    
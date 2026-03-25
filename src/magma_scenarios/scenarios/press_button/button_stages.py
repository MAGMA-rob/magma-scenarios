
from magma_core.base.data_structures import Instruction, Log, Situation
from magma_core.base.stage import BaseTaskStage, ConstraintBaseStage
from magma_core.base.goals import BaseGoal

from .helper import attributes, tensor_is_button_pressed

import torch
from typing import List, Dict

class CountPressed(BaseGoal):

    def __init__(self, btn_list : List[str], nb : int) -> None:
        super().__init__("Count Pressed", f"btns={btn_list}")
        self.nb = nb
        self.button_names = btn_list

    def verify(self, obs: Dict) -> torch.Tensor:
        device = obs["extra"][self.button_names[0]].device
        batch_size = obs["extra"][self.button_names[0]].shape[0]

        pressed = torch.zeros(batch_size, device=device, dtype=torch.int32)
        err = torch.zeros(batch_size, device=device, dtype=torch.bool)
        for obj_name, pose in obs["extra"].items():
            if "sw" not in obj_name:
                continue
            is_pressed = tensor_is_button_pressed(pose[:, -1])

            if obj_name in self.button_names:
                pressed += is_pressed.int()
            else:
                err |= is_pressed

        out = (pressed == self.nb).int()
        out[err] = -1
        return out.int()

class Pressed(BaseGoal):

    def __init__(self, btn : str) -> None:
        super().__init__("Pressed", f"btn={btn}")
        self.btn = btn

    def verify(self, obs: Dict) -> torch.Tensor:
        return tensor_is_button_pressed(obs["extra"][self.btn][:, -1])

class PressMultipleButton(BaseTaskStage):

    target_steps = 1
    acceptance_steps = 0

    def __init__(self, n : int, all_button : List[str], instruction : Instruction, reset_at_end: bool = False) -> None:
        super().__init__([CountPressed(all_button,n)],reset_at_end, f"The goal of this stage is to have {n} button pressed from the list : {all_button}")
        self.situation = Situation(
            memory=[],
            preserved_memory_indices=[],
            attributes=attributes,
            instruction=instruction,
            flag_answer_to_user= n == len(all_button)
        )
    
class PressConstraintStage(ConstraintBaseStage):

    def __init__(self, constraint: str) -> None:
        super().__init__(constraint, [], attributes)

class PressButton(BaseTaskStage):

    target_steps = 1
    acceptance_steps = 0

    def __init__(self, button : str, instruction : Instruction, last : bool = False) -> None:
        super().__init__([Pressed(button)],True, f"The goal of this stage is to press {button}")
        self.button = button
        self.situation = Situation(
            memory=[],
            preserved_memory_indices=[],
            attributes=attributes,
            instruction=instruction,
            flag_answer_to_user= last
        )
        
    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if len(stage_log) == 0: return 0
        if stage_log[-1].content == self.button: return 1
        return -1

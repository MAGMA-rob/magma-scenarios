
from magma_core.simulation.data_structures import Instruction, Log, StageInput
from magma_core.simulation.stage import BaseTaskStage, StageErrorParameters, StageGlobalParameters
from magma_core.simulation.goals import BaseGoal

from .helper import tensor_is_button_pressed
from magma_scenarios.templates.errors import OneShotToolFailureError

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
    """
    Each buttons can be only pressed once,
    as we are verifying all buttons state simustaneously
    """

    target_tool_calls = 1
    max_tool_calls = 1

    def __init__(
            self,
            n : int,
            all_button : List[str],
            instruction : Instruction,
            reset_at_end: bool = False
        ) -> None:
        super().__init__(
            goals=[CountPressed(all_button,n)],
            stage_goal_description=f"The goal of this stage is to have {n} button pressed from the list : {all_button}",
            stage_input=StageInput(
                instruction=instruction,
                flag_answer_to_user= (n == len(all_button))
            ),
            global_parameters=StageGlobalParameters(
                reset_at_end
            ),
            error_parameters=StageErrorParameters(
                possible_errors=[OneShotToolFailureError()]
            ),
        )
        self.n = n
        self.all_button = list(all_button)

    def _to_spec_arguments(self) -> Dict:
        return {
            "n": self.n,
            "all_button": self.all_button.copy(),
            "instruction": self.get_stage_input().instruction,
            "reset_at_end": self.should_reset_at_end(),
        }

class PressButton(BaseTaskStage):

    target_tool_calls = 1
    max_tool_calls = 1

    def __init__(
            self,
            button : str,
            instruction : Instruction,
            last : bool = False,
            target_tool_calls: int = 1,
        ) -> None:
        self.target_tool_calls = target_tool_calls
        self.max_tool_calls = target_tool_calls
        super().__init__(
            [Pressed(button)],
            f"The goal of this stage is to press {button}",
            StageInput(instruction,last),
            StageGlobalParameters(True),
            error_parameters=StageErrorParameters(
                possible_errors=[OneShotToolFailureError()]
            ),
        )
        self.button = button

    def _to_spec_arguments(self) -> Dict:
        return {
            "button": self.button,
            "instruction": self.get_stage_input().instruction,
            "last": self.get_stage_input().flag_answer_to_user,
            "target_tool_calls": self.target_tool_calls,
        }
        
    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if len(stage_log) == 0: return 0
        if stage_log[-1].content == self.button: return 1
        return -1

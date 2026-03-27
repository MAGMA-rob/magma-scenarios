from magma_core.base.errors import BaseError
from torch._tensor import Tensor
import torch
from typing import Any, List, Dict

from magma_core.base.stage import BaseTaskStage
from magma_core.base.data_structures import Instruction, Log, Situation
from magma_core.utils.env_utils import is_object_inside_target
from magma_core.base.goals import BaseGoal, At, MaxAt

from .color_sorting_errors import MaskRemainingCubesError
from .attributes import att

class ExactCubeAt(At):

    def __init__(self, obj_name: str, location: str, desired_nb : int, thresh: float = 0.1):
        super().__init__(obj_name, location, thresh)
        self.desired_nb = desired_nb

    def verify(self, obs: Dict) -> Tensor:
        N = obs["extra"]["agent_tcp"].shape[0]
        device = obs["extra"]["agent_tcp"].device
        nb = torch.zeros(N, device=device)

        for obj_name, obj_pose in obs["extra"].items():
            if self.obj in obj_name:
                nb += is_object_inside_target(obj_pose,obs["extra"][self.location],keep_tensor=True).int()
            
        out = torch.zeros(N, device=device)
        out[(nb == self.desired_nb)] = 1
        return out

class MaxSortedColor(MaxAt):

    def __init__(self, color : str, maximum: int):
        super().__init__([f"{color}_cube_{i+1}" for i in range(3)], f"{color}_box_pose", maximum, True)

class CountCubes(BaseGoal):

    def __init__(self, desired_nb : int) -> None:
        super().__init__("Count cubes correctly placed in boxes", f"desired_nb={desired_nb}")
        self.nb = desired_nb

    def verify(self, obs: Dict) -> Tensor:
        N = obs["extra"]["agent_tcp"].shape[0]
        device = obs["extra"]["agent_tcp"].device
        nb_yellow = torch.zeros(N, device=device)
        nb_green = torch.zeros(N, device=device)

        for obj_name, obj_pose in obs["extra"].items():
            if "green_cube" in obj_name:
                nb_green += is_object_inside_target(obj_pose,obs["extra"]["green_box_pose"],keep_tensor=True).int()
            elif "yellow_cube" in obj_name:
                nb_yellow += is_object_inside_target(obj_pose,obs["extra"]["yellow_box_pose"],keep_tensor=True).int()
                
        out = torch.zeros(N, device=device)
        out[((nb_yellow + nb_green) >= self.nb)] = 1
        return out

class SortByColorStage(BaseTaskStage):

    target_steps = 3
    acceptance_steps = 1

    possible_errors = [MaskRemainingCubesError()]

    def __init__(self, instruction : Instruction, nb_good_place : int, last : bool, max_yellow : int = 3, max_green : int = 3) -> None:            
        self.situation = Situation(
            memory=[],
            preserved_memory_indices=[],
            attributes=att,
            flag_answer_to_user=last,
            instruction=instruction
        )
        
        super().__init__(
            [CountCubes(nb_good_place),MaxSortedColor("green",max_green),MaxSortedColor("yellow",max_yellow)],
            reset_at_end=last,
            stage_goal_description=f"The goal of this stage is to have {nb_good_place} cubes correctly placed in their designated color box."
        )

class ExactSortByColorStage(BaseTaskStage):

    target_steps = 3
    acceptance_steps = 1

    def __init__(self, instruction : Instruction, nb_green : int, nb_yellow : int, last : bool) -> None:            
        self.situation = Situation(
            memory=[],
            preserved_memory_indices=[],
            attributes=att,
            flag_answer_to_user=last,
            instruction=instruction
        )
        
        super().__init__(
            [
                ExactCubeAt("green", "green_box_pose", nb_green),
                ExactCubeAt("yellow", "yellow_box_pose", nb_yellow),
                MaxSortedColor("green", nb_green),
                MaxSortedColor("yellow", nb_yellow),
            ],
            reset_at_end=last,
            stage_goal_description=f"The goal of this stage is to have {nb_green} green cubes in green box and {nb_yellow} in yellow box."
        )

class DetectionStage(BaseTaskStage):

    target_steps = 1
    acceptance_steps = 0

    def __init__(self, reset_at_end: bool, instruction : Instruction) -> None:
        super().__init__([], reset_at_end, "The goal of this stage is to call the detection function to ensure that the original task have been correctly completed")

        self.situation = Situation(
            memory=[],
            preserved_memory_indices=[],
            attributes=att,
            flag_answer_to_user=True,
            instruction=instruction
        )

    def verif_log_completion(self, stage_log: List[Log], full_log: List[Log]) -> int:
        if len(stage_log) == 0:
            return 0
        for l in stage_log:
            if l.function != "get_object_state":
                return -1
        return 1
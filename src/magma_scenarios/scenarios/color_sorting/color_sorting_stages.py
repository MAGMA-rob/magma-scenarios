from magma_core.base.errors import BaseError
from torch._tensor import Tensor
import torch
from typing import List, Dict

from magma_core.base.stage import BaseTaskStage
from magma_core.base.data_structures import Instruction, Log, Situation
from magma_core.utils.env_utils import is_object_inside_target
from magma_core.base.goals import BaseGoal, ExactCountAt, MaxAt

from .color_sorting_errors import MaskRemainingCubesError


class MaxSortedColor(MaxAt):

    def __init__(self, color : str, maximum: int):
        super().__init__([f"{color}_cube_{i+1}" for i in range(3)], f"{color}_box_pose", maximum, True)

class CountCubes(BaseGoal):

    def __init__(self, desired_nb : int, colors : List[str], inverted : bool = False) -> None:
        super().__init__("Count cubes correctly placed in boxes", f"desired_nb={desired_nb}")
        self.nb = desired_nb
        self.colors = colors.copy()
        self.cubes_color = colors.copy()
        if inverted:
            self.cubes_color.reverse()

    def verify(self, obs: Dict) -> Tensor:
        N = obs["extra"]["agent_tcp"].shape[0]
        device = obs["extra"]["agent_tcp"].device
        nb = torch.zeros(N, device=device)

        for obj_name, obj_pose in obs["extra"].items():
            for i in range(2):
                if f"{self.cubes_color[i]}_cube" in obj_name:
                    nb += is_object_inside_target(
                        obj_pose,
                        obs["extra"][f"{self.colors[i]}_box_pose"],keep_tensor=True
                    ).int()
                    break
            
        out = torch.zeros(N, device=device)
        out[(nb >= self.nb)] = 1
        return out
    
class ExactCountCube(ExactCountAt):

    def __init__(self, color : str, location: str, expected: int):
        objects = [f"{color}_cube_{i+1}"for i in range(3)]
        super().__init__(objects, location, expected)

class SortByColorStage(BaseTaskStage):

    target_steps = 3
    acceptance_steps = 1

    possible_errors = [MaskRemainingCubesError()]

    def __init__(
            self,
            instruction : Instruction,
            nb_good_place : int,
            attributes : Dict,
            last : bool, 
            max_assignment : Dict = {}
        ) -> None:            
        self.situation = Situation(
            memory=[],
            preserved_memory_indices=[],
            attributes=attributes,
            flag_answer_to_user=last,
            instruction=instruction
        )

        g : List[BaseGoal] = [CountCubes(nb_good_place,colors=attributes["known_box_color"])]
        for color, max_nb in max_assignment.items():
            g.append(MaxSortedColor(color,max_nb))
        
        super().__init__(
            g,
            reset_at_end=last,
            stage_goal_description=f"The goal of this stage is to have {nb_good_place} cubes correctly placed in their designated color box."
        )

class ExactSortByColorStage(BaseTaskStage):

    target_steps = 3
    acceptance_steps = 1

    def __init__(self, instruction : Instruction, attributes : Dict, assignment : Dict, last : bool) -> None:            
        self.situation = Situation(
            memory=[],
            preserved_memory_indices=[],
            attributes=attributes,
            flag_answer_to_user=last,
            instruction=instruction
        )
        if assignment == {}:
            raise RuntimeError("Got an empty assignment")
        
        msg = []
        g : List[BaseGoal] = []
        for color, nb in assignment.items():
            g.append(
                ExactCountCube(color,f"{color}_box_pose",nb)
            )
            msg.append(f"{nb} {color} cube in {color} box")
        
        super().__init__(
            g,
            reset_at_end=last,
            stage_goal_description=f"The goal of this stage is to have: {' and '.join(msg)}."
        )

class DetectionStage(BaseTaskStage):

    target_steps = 1
    acceptance_steps = 0

    def __init__(self, reset_at_end: bool, attributes : Dict, instruction : Instruction) -> None:
        super().__init__([], reset_at_end, "The goal of this stage is to call the detection function to ensure that the original task have been correctly completed")

        self.situation = Situation(
            memory=[],
            preserved_memory_indices=[],
            attributes=attributes,
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
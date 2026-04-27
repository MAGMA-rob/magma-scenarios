import random
from typing import Dict, Any, List, Optional
import torch
from magma_core.base.data_structures import Observation
from magma_core.base.data_structures.tools import ToolResult
from magma_core.utils.env_utils import is_object_inside_target
from magma_scenarios.templates.errors import MaskedObjectError, GraspFailureError
from magma_scenarios.utils import sapien_to_tensor
from torch import rand
from .attributes import people, teams,att


def _get_available_capsules(obs: Observation, env_id: int) -> List[str]:
    remaining = []
    extra = obs.maniskill_obs["extra"]
    capsule_target = sapien_to_tensor(obs.add_constants["loaded_capsule_pose"],extra["coffee_maker"].device)
    target_absolue = torch.add(extra["coffee_maker"][env_id][:7],capsule_target)
    for obj_name, obj_data in extra.items():
        if not any(pod in obj_name for pod in att["coffee_pod"]):
            continue
        if not is_object_inside_target(
            obj_data["pose"][env_id],
            target_absolue
        ):
            remaining.append(obj_name)
    return remaining

class GraspCapsuleFailureError(GraspFailureError):
    recovery_extra_steps = 1

    def __init__(self,max_impossible= 1) -> None:
        super().__init__()
        self.max_nb = max_impossible

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        remaining = _get_available_capsules(obs,env_id)
        if len(remaining) <= 1:
            return {
                "inaccessible" : [],
            }

        if len(remaining) == 2:
            nb = 1
        else:
            nb = random.randint(1,self.max_nb) 
        return {"inaccessible" : random.sample(remaining,k=nb)}

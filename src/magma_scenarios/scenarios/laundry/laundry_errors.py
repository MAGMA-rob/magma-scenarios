import random
from typing import Dict, Any, List, Optional

from magma_core.base.data_structures import Observation
from magma_core.base.data_structures.tools import ToolResult
from magma_core.utils.env_utils import is_object_inside_target
from magma_scenarios.templates.errors import MaskedObjectError, GraspFailureError
from torch import rand
from .attributes import all_detergents, all_clothes


def _get_unwashed_clothes(obs: Observation, env_id: int) -> List[str]:
    remaining = []
    extra = obs.maniskill_obs["extra"]
    for obj_name, obj_data in extra.items():
        if obj_name in all_detergents or obj_name in ["washing_machine_basket", "agent_tcp", "washing_machine"]:
            continue
        if obj_name not in all_clothes:
            continue
        if not is_object_inside_target(
            obj_data["pose"][env_id],
            extra["washing_machine_basket"]["pose"][env_id]
        ):
            remaining.append(obj_name)
    return remaining

class GraspClothesFailureError(GraspFailureError):
    recovery_extra_steps = 1

    def __init__(self,max_impossible= 1) -> None:
        super().__init__()
        self.max_nb = max_impossible

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        remaining = _get_unwashed_clothes(obs,env_id)
        if len(remaining) <= 1:
            return {
                "inaccessible" : [],
            }

        if len(remaining) == 2:
            nb = 1
        else:
            nb = random.randint(1,self.max_nb) 
        return {"inaccessible" : random.sample(remaining,k=nb)}

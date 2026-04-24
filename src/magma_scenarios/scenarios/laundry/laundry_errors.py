import random
from typing import Dict, Any, List

from magma_core.base.data_structures import Observation
from magma_core.utils.env_utils import is_object_inside_target
from magma_scenarios.templates.errors import GraspFailureError
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

    def __init__(self, all_requested_objects : List[str] = [], max_impossible = 2) -> None:
        super().__init__()
        self.max_nb = max_impossible
        self.all_requested_objects = all_requested_objects

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:        
        remaining = _get_unwashed_clothes(obs,env_id)
        can_be_masked = [r for r in remaining if r in self.all_requested_objects]
        if len(can_be_masked) <= 1:
            return {
                "inaccessible" : [],
            }

        if len(can_be_masked) == 2:
            nb = 1
        else:
            nb = random.randint(1,self.max_nb)
        return {"inaccessible" : random.sample(can_be_masked,k=nb)}

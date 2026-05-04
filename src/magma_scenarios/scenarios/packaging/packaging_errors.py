from re import M
from magma_core.base.data_structures.observation import Observation
from magma_core.base.data_structures.tools import ToolResult
from magma_scenarios.templates.errors import MaskedObjectError, GraspFailureError
from typing import Dict, Optional, Any, List
from .attributes import fruits, drinks, main_course
from magma_core.utils.env_utils import is_object_inside_target
import random

def _get_remaining_food(obs : Observation, env_id : int) -> List[str]:
    remaining = []
    extra = obs.maniskill_obs["extra"]
    for obj_name, obj_data in extra.items():
        if obj_name in ["tray", "agent_tcp"] :
            continue
        if obj_name not in [*fruits,*drinks,*main_course] :
            continue

        remaining.append(obj_name)
    return remaining



class GraspFoodFailureError(GraspFailureError):
    recovery_extra_steps = 1
    def __init__(self,all_requested_objects : List[str] = [],max_masking = 2) -> None:
        super().__init__()
        self.max_nb = max_masking
        self.all_requested_objects = all_requested_objects

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        remaining = _get_remaining_food(obs,env_id)
        can_be_masked = [r for r in remaining if r in self.all_requested_objects]
        if len(can_be_masked) <= 1 :
            return{
                "inaccessible" : []
            }

        if len(can_be_masked) == 2 :
            nb = 1
        else:
            nb = random.randint(1,self.max_nb)
        return {"inaccessible" : random.sample(can_be_masked, k = nb)}

class MaskFoodError(MaskedObjectError):
    def __init__(self,all_requested_objects : List[str] = [],max_masking = 2) -> None:
        super().__init__(tool_execution_target_key="target_name")
        self.max_nb = max_masking
        self.all_requested_objects = all_requested_objects

    def initialize(self, obs: Observation, env_id: int) -> Optional[Dict[str, Any]]:
        remaining = _get_remaining_food(obs,env_id)
        can_be_masked = [r for r in remaining if r in self.all_requested_objects]
        if len(can_be_masked) <= 1 :
            return{
                "masked" : []
            }

        if len(can_be_masked) == 2 :
            nb = 1
        else:
            nb = random.randint(1,self.max_nb)
        return {"masked" : random.sample(can_be_masked, k = nb)}
    
    def apply_post_verif(self, tool_result, arguments):
        if not tool_result.context:
            return

        masked = arguments.get("masked", [])

        table = tool_result.context.get("table", {})
        tray = tool_result.context.get("tray", {})

        for m in masked:
            table.pop(m, None)
            tray.pop(m, None)

        s = "this is the position of existing object : "

        if len(table) == 0:
            s += "the table is empty"
        else:
            s += "the table contain " + ", ".join(table.keys())

        if len(tray) == 0:
            s += ", and the tray is empty"
        else:
            s += ", and the tray contain " + ", ".join(tray.keys())

        tool_result.reason = s
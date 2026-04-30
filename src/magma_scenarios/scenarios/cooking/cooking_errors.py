from re import M
from magma_core.base.data_structures.observation import Observation
from magma_core.base.data_structures.tools import ToolResult
from magma_scenarios.templates.errors import MaskedObjectError
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
        if not is_object_inside_target(
            obj_data[env_id],
            extra["tray"][env_id],
            thresh= 0.2
        ):
            remaining.append(obj_name)
    return remaining



class GraspFoodFailureError(MaskedObjectError):
    def __init__(self,all_requested_objects : List[str] = [],max_masking = 2) -> None:
        super().__init__(tool_execution_target_key="target_name")
        self.max_nb = max_masking
        self.all_requested_objects = all_requested_objects

    def initialize(self, obs: Observation, env_id: int) -> Optional[Dict[str, Any]]:
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
                "inaccessible" : []
            }

        if len(can_be_masked) == 2 :
            nb = 1
        else:
            nb = random.randint(1,self.max_nb)
        return {"inaccessible" : random.sample(can_be_masked, k = nb)}
    
    def apply_post_verif(self, tool_result, arguments):
        if not tool_result.context:
            return

        visible = list(tool_result.context.get("visible_food", []))  

        masked = arguments.get("inaccessible", [])

        for m in masked:
            if m in visible:
                visible.remove(m)

        tool_result.reason = f"I see: {visible}"
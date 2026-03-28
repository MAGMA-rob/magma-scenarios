import random
from typing import Dict, Any, Optional

from magma_core.base.data_structures import Observation
from magma_core.base.data_structures.tools import ToolExecution, ToolResult
from magma_core.base.errors import BaseError
from magma_core.utils.env_utils import is_object_inside_target

class MaskRemainingCubesError(BaseError):

    recovery_extra_steps = 0

    def __init__(self, max_masking = 2) -> None:
        super().__init__()
        self.max_nb = max_masking
        # je pourrai juste demander à ce que les classes enfant défine une fonction
        # qui retournerai la liste des objets possible à masquer.
        # le reste pourrait être une logique commune aux erreurs de perception

    def initialize(self, obs : Observation, env_id : int) -> Optional[Dict[str, Any]]:
        remaining = []
        for obj_name, obj_pose in obs.maniskill_obs['extra'].items():
            if "cube" in obj_name:
                if (not is_object_inside_target(
                        obj_pose[env_id],
                        obs.maniskill_obs["extra"]["green_box_pose"][env_id]
                    )
                    and not is_object_inside_target(
                        obj_pose[env_id],
                        obs.maniskill_obs["extra"]["yellow_box_pose"][env_id])
                    ):
                    remaining.append(obj_name)

        if len(remaining) <= 1:
            return {
                "masked" : []
            }
        
        if len(remaining) == 2:
            nb = 1
        else:
            nb = random.randint(1,self.max_nb)
        
        masked = random.sample(remaining,k=nb)
        print(masked)
        return {
            "masked" : masked
        }

    def apply_pre_exec(self, tool_execution: ToolExecution, arguments: Dict[str, Any]):
        masked = arguments.get("masked",None)
        if masked is None or len(masked)==0:
            return
        target_name = tool_execution.context.get("target_name", None)
        if target_name is None:
            return

        if target_name in masked:
            tool_execution.fail(
                f"Unknown object: {target_name}. Please use only detected objects."
            )

    def apply_post_verif(self, tool_result: ToolResult, arguments: Dict[str, Any]):
        if not tool_result.context:
            return

        remaining_objects = tool_result.context.get("table",[])
        if len(remaining_objects) <= 1:
            return

        masked = arguments.get("masked",[])

        if len(masked) > 0:
            for m in masked:
                remaining_objects.remove(m)

            s = "This is the position of existing objects: "
            if len(tool_result.context['green_box']) == 0:
                s += "green_box is empty. "
            else:
                s += ",".join(tool_result.context["green_box"]) + " are in the green_box. "

            if len(tool_result.context['yellow_box']) == 0:
                s += "green_box is empty. "
            else:
                s += ",".join(tool_result.context['yellow_box']) + " are in the yellow_box. "

            if len(remaining_objects) > 0:
                s += ",".join(remaining_objects) + " are not sorted."

            tool_result.reason = s

    def get_description(self, arguments: Dict[str, Any] | None) -> str:
        if arguments is None or arguments.get("masked") is None:
            return "Mask some remaining cubes from perception."
        return f"Masked cubes from perception: {arguments['masked']}."

class GraspFailureError(BaseError):

    recovery_extra_steps = 1

    def __init__(self, max_impossible = 1) -> None:
        super().__init__()
        self.max_nb = max_impossible

    def initialize(self, obs : Observation, env_id : int) -> Optional[Dict[str, Any]]:
        remaining = []
        for obj_name, obj_pose in obs.maniskill_obs['extra'].items():
            if "cube" in obj_name:
                if (not is_object_inside_target(
                        obj_pose[env_id],
                        obs.maniskill_obs["extra"]["green_box_pose"][env_id]
                    )
                    and not is_object_inside_target(
                        obj_pose[env_id],
                        obs.maniskill_obs["extra"]["yellow_box_pose"][env_id])
                    ):
                    remaining.append(obj_name)

        if len(remaining) <= 1:
            return {
                "innaccessible" : []
            }
        
        if len(remaining) == 2:
            nb = 1
        else:
            nb = random.randint(1,self.max_nb)
        
        impossible_to_grasp = random.sample(remaining,k=nb)
        print(impossible_to_grasp)
        return {
            "innaccessible" : impossible_to_grasp
        }

    def apply_pre_exec(self, tool_execution: ToolExecution, arguments: Dict[str, Any]):
        innaccessible = arguments.get("innaccessible",None)
        if innaccessible is None or len(innaccessible)==0:
            return
        target_name = tool_execution.context.get("target_name", None)
        if target_name is None:
            return

        if target_name in innaccessible:
            tool_execution.fail(
                f"Failed to grasp: {target_name}. The object is unreachable right now."
            )

    def get_description(self, arguments: Dict[str, Any] | None) -> str:
        if arguments is None or arguments.get("masked") is None:
            return "Make some object impossible to take"
        return f"These objects are impossible to take right now: {arguments['masked']}."

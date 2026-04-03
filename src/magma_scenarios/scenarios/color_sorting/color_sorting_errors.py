import random
from typing import Dict, Any, List, Optional

from magma_core.base.data_structures import Observation
from magma_core.base.data_structures.tools import ToolResult
from magma_core.utils.env_utils import is_object_inside_target
from magma_scenarios.templates.errors import MaskedObjectError, GraspFailureError

# On pourrait ajouter une erreur en mode boite inaccessible ou boite non détecté.

def _get_scene_colors(obs: Observation) -> List[str]:
    colors = obs.add_constants.get("colors", obs.task_attributes.get("known_box_color", None))
    if colors is None:
        raise RuntimeError(
            "Color sorting errors require colors in obs.add_constants['colors'] "
            "or obs.task_attributes['known_box_color']."
        )
    return list(colors)

def _get_remaining_cubes(obs: Observation, env_id: int, colors: List[str]) -> List[str]:
    remaining = []
    for obj_name, obj_pose in obs.maniskill_obs["extra"].items():
        if "cube" not in obj_name:
            continue

        if all(
            not is_object_inside_target(
                obj_pose[env_id],
                obs.maniskill_obs["extra"][f"{color}_box_pose"][env_id]
            )
            for color in colors
        ):
            remaining.append(obj_name)
    return remaining

def _build_detection_reason(context: Dict[str, Any], colors: List[str], remaining_objects: List[str]) -> str:
    s = "This is the position of existing objects: "
    for color in colors:
        key = f"{color}_box"
        if len(context[key]) == 0:
            s += f"{color}_box is empty. "
        else:
            s += ",".join(context[key]) + f" are in the {color}_box. "

    if len(remaining_objects) > 0:
        s += ",".join(remaining_objects) + " are not sorted."

    return s

class MaskRemainingCubesError(MaskedObjectError):

    def __init__(self, max_masking = 2) -> None:
        super().__init__(tool_execution_target_key="target_name")
        self.max_nb = max_masking

    def initialize(self, obs : Observation, env_id : int) -> Optional[Dict[str, Any]]:
        colors = _get_scene_colors(obs)
        remaining = _get_remaining_cubes(obs, env_id, colors)

        if len(remaining) <= 1:
            return {
                "masked" : [],
                "colors": colors,
            }
        
        if len(remaining) == 2:
            nb = 1
        else:
            nb = random.randint(1,self.max_nb)
        
        masked = random.sample(remaining,k=nb)
        return {
            "masked" : masked,
            "colors": colors,
        }

    def apply_post_verif(self, tool_result: ToolResult, arguments: Dict[str, Any]):
        if not tool_result.context:
            return

        remaining_objects = list(tool_result.context.get("table",[]))
        if len(remaining_objects) <= 1:
            return

        masked = arguments.get("masked",[])
        colors = arguments.get("colors", [])

        if len(masked) > 0:
            for m in masked:
                if m in remaining_objects:
                    remaining_objects.remove(m)

            tool_result.reason = _build_detection_reason(
                tool_result.context,
                colors,
                remaining_objects,
            )

class GraspCubeFailureError(GraspFailureError):

    recovery_extra_steps = 1

    def __init__(self, max_impossible = 1) -> None:
        super().__init__()
        self.max_nb = max_impossible

    def initialize(self, obs : Observation, env_id : int) -> Dict[str, Any]:
        colors = _get_scene_colors(obs)
        remaining = _get_remaining_cubes(obs, env_id, colors)

        if len(remaining) <= 1:
            return {
                "innaccessible" : [],
                "colors": colors,
            }
        
        if len(remaining) == 2:
            nb = 1
        else:
            nb = random.randint(1,self.max_nb)
        
        impossible_to_grasp = random.sample(remaining,k=nb)
        return {
            "innaccessible" : impossible_to_grasp,
            "colors": colors,
        }

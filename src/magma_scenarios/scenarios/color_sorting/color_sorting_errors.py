import random
from itertools import combinations
from typing import Any, Dict, List, Optional

from magma_core.base.data_structures import Observation
from magma_core.base.data_structures.tools import ToolResult
from magma_core.utils.env_utils import is_object_inside_target
from magma_scenarios.templates.errors import GraspFailureError, MaskedObjectError

# On pourrait ajouter une erreur en mode boite inaccessible ou boite non détecté.


def _get_scene_colors(obs: Observation) -> List[str]:
    colors = obs.add_constants.get("colors", obs.task_attributes.get("known_box_color", None))
    if colors is None:
        raise RuntimeError(
            "Color sorting errors require colors in obs.add_constants['colors'] "
            "or obs.task_attributes['known_box_color']."
        )
    return list(colors)


def _sample_error_objects(
        obs: Observation,
        env_id: int,
        colors: List[str],
        assignment: Dict[str, int],
        max_nb: int,
    ) -> List[str]:
    remaining_by_color = {color: [] for color in colors}
    sorted_counts = {color: 0 for color in colors}

    if assignment == {}:
        assignment = {color : 3 for color in colors}

    out = []

    for obj_name, obj_pose in obs.maniskill_obs["extra"].items():
        if "cube" in obj_name:
            for color in colors:
                if color in obj_name:
                    if is_object_inside_target(obj_pose[env_id], obs.maniskill_obs["extra"][f"{color}_box_pose"][env_id]):
                        sorted_counts[color] += 1
                    else:
                        remaining_by_color[color].append(obj_name)
                    break
    
    needed_total = 0
    slack_per_color : Dict[str,int] = {}
    for c in colors:
        need = max(assignment[c] - sorted_counts[c],0)
        needed_total += need
        if need > 0:
            slack_per_color[c] = len(remaining_by_color[c]) - need

    if needed_total <= 1:
        color = None
        for c in colors:
            if assignment[c] - sorted_counts[c] == 1:
                color = c
                break
        if color is None:
            raise RuntimeError("Impossible fail")
        
        if len(remaining_by_color[color]) <= 1:
            return []
        nb = min(max_nb,len(remaining_by_color[color])-1)
        return random.sample(remaining_by_color[color],k=nb)
    
    if len(slack_per_color) == 1: #case where only one color is interesting
        for c in slack_per_color.keys():
            n = len(remaining_by_color[c])
            random.shuffle(remaining_by_color[c])
            for obj in remaining_by_color[c]:
                if max_nb <= 0 or n == 1:
                    break
                out.append(obj)
                max_nb-=1
                n-=1
        
    else:
        # case where multiple color can be used
        selected_color = random.choice(list(slack_per_color.keys()))
        random.shuffle(remaining_by_color[selected_color])
        out.extend(remaining_by_color[selected_color][:max_nb])

    return out


class MaskRemainingCubesError(MaskedObjectError):

    def __init__(
            self,
            max_masking: int = 2,
            assignment: Optional[Dict[str, int]] = None,
        ) -> None:
        super().__init__(tool_execution_target_key="target_name")
        self.max_nb = max_masking
        self.assignment = {} if assignment is None else assignment.copy()


    def initialize(self, obs : Observation, env_id : int) -> Optional[Dict[str, Any]]:
        colors = _get_scene_colors(obs)
        d = {
            "masked": _sample_error_objects(
                obs,
                env_id,
                colors,
                self.assignment,
                self.max_nb,
            )
        }
        return d

    def apply_post_verif(self, tool_result: ToolResult, arguments: Dict[str, Any]):
        if not tool_result.context:
            return

        remaining_objects = list(tool_result.context.get("table", []))
        masked = arguments.get("masked", [])
        if len(remaining_objects) <= 1 or len(masked) == 0:
            return

        for m in masked:
            if m in remaining_objects:
                remaining_objects.remove(m)

        reason = "This is the position of existing objects: "
        for key in tool_result.context:
            if key == "table":
                continue
            if len(tool_result.context[key]) == 0:
                reason += f"{key} is empty. "
            else:
                reason += ",".join(tool_result.context[key]) + f" are in the {key}. "

        if len(remaining_objects) > 0:
            reason += ",".join(remaining_objects) + " are not sorted."

        tool_result.reason = reason


class GraspCubeFailureError(GraspFailureError):

    recovery_extra_steps = 1

    def __init__(
            self,
            max_impossible: int = 1,
            assignment: Optional[Dict[str, int]] = None
        ) -> None:
        super().__init__()
        self.max_nb = max_impossible
        self.assignment = {} if assignment is None else assignment.copy()

    def initialize(self, obs : Observation, env_id : int) -> Dict[str, Any]:
        colors = _get_scene_colors(obs)
        
        d = {
            "inaccessible": _sample_error_objects(
                obs,
                env_id,
                colors,
                self.assignment,
                self.max_nb,
            )
        }
        print(d)
        return d

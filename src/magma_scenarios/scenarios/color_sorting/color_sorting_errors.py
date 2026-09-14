import copy
import random
from typing import Any, Dict, List, Optional

from magma_core.simulation.data_structures import Observation
from magma_core.simulation.data_structures.tools import ToolResult
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_scenarios.templates.errors import GraspFailureError, MaskedObjectError


def _sample_error_objects(
    obs: Observation,
    env_id: int,
    assignment: Dict[str, Dict[str, int]],
    minimum: int,
    max_nb: int,
    thresh: float,
) -> List[str]:
    """Select movable instances while preserving enough direct moves to finish."""
    if max_nb <= 0:
        return []

    extra = obs.maniskill_obs["extra"]
    tray_names = [name for name in extra if name.endswith("_tray")]
    current = {
        location: {color: 0 for color in color_counts}
        for location, color_counts in assignment.items()
    }
    objects_by_cell: Dict[tuple[str, str], List[str]] = {}

    for object_name, object_pose in extra.items():
        if "_cube_" not in object_name:
            continue

        color = object_name.split("_cube_", 1)[0]
        location = "table"
        for tray_name in tray_names:
            if is_object_inside_target(
                object_pose[env_id],
                extra[tray_name][env_id],
                thresh=thresh,
                keep_tensor=False,
            ):
                location = tray_name
                break

        if location in current and color in current[location]:
            current[location][color] += 1
            objects_by_cell.setdefault((location, color), []).append(object_name)

    matching_count = 0
    colors_with_deficit = set()
    surplus_cells = set()

    for location, color_counts in assignment.items():
        for color, target_count in color_counts.items():
            current_count = current.get(location, {}).get(color, 0)
            matching_count += min(current_count, target_count)

            # A useful direct move must end in a tray: the current put tool does
            # not support dropping an object back onto the table.
            if location != "table" and current_count < target_count:
                colors_with_deficit.add(color)

            if current_count > target_count:
                surplus_cells.add((location, color))

    # SortByColorStage instances are progressive: stage n starts with n - 1
    # matching cubes and needs exactly one useful move. If that invariant is
    # not met, injecting no error is safer than guessing which path is valid.
    if matching_count != minimum - 1:
        return []

    candidates = [
        object_name
        for cell in surplus_cells
        if cell[1] in colors_with_deficit
        for object_name in objects_by_cell.get(cell, [])
    ]
    max_safe = min(max_nb, len(candidates) - 1)
    if max_safe <= 0:
        return []

    return random.sample(
        candidates,
        k=random.randint(1, max_safe),
    )


class MaskRemainingCubesError(MaskedObjectError):

    def __init__(
        self,
        max_masking: int = 0,
        assignment: Optional[Dict[str, Dict[str, int]]] = None,
        minimum: int = 0,
        thresh: float = 0.3,
    ) -> None:
        super().__init__(tool_execution_target_key="target_name")
        self.max_nb = max_masking
        self.assignment = copy.deepcopy(assignment) if assignment is not None else None
        self.minimum = minimum
        self.thresh = thresh

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        if self.assignment is None:
            return {"masked": []}

        return {
            "masked": _sample_error_objects(
                obs,
                env_id,
                self.assignment,
                self.minimum,
                self.max_nb,
                self.thresh,
            )
        }

    def apply_post_verif(self, tool_result: ToolResult, arguments: Dict[str, Any]):
        if not tool_result.context:
            return

        masked = set(arguments.get("masked", []))
        if not masked:
            return

        visible_by_location = {
            location: [name for name in object_names if name not in masked]
            for location, object_names in tool_result.context.items()
        }
        descriptions = []
        for location, object_names in visible_by_location.items():
            if object_names:
                descriptions.append(f"{location} contains {', '.join(object_names)}")
            else:
                descriptions.append(f"{location} contains no cube")

        tool_result.context = visible_by_location
        tool_result.reason = ". ".join(descriptions) + "."


class GraspCubeFailureError(GraspFailureError):

    def __init__(
        self,
        max_impossible: int = 0,
        assignment: Optional[Dict[str, Dict[str, int]]] = None,
        minimum: int = 0,
        thresh: float = 0.3,
    ) -> None:
        super().__init__()
        self.max_nb = max_impossible
        self.assignment = copy.deepcopy(assignment) if assignment is not None else None
        self.minimum = minimum
        self.thresh = thresh

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        if self.assignment is None:
            return {"inaccessible": []}

        return {
            "inaccessible": _sample_error_objects(
                obs,
                env_id,
                self.assignment,
                self.minimum,
                self.max_nb,
                self.thresh,
            )
        }

import copy
import random
from typing import Any, Dict, List, Optional

from magma_core.simulation.data_structures import Observation, ToolResult
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_scenarios.templates.errors import GraspFailureError, MaskedObjectError

def _object_type(object_name: str) -> str:
    return object_name.rsplit("_", 1)[0]


def _sample_error_objects(
    obs: Observation,
    env_id: int,
    required_type_counts: Dict[str, Dict[str, int]],
    fixed_targets_by_object: Dict[str, str],
    minimum: int,
    max_nb: int,
    thresh: float,
) -> List[str]:
    """Select useful moves while preserving one direct way to advance."""
    if max_nb <= 0:
        return []

    extra = obs.maniskill_obs["extra"]
    locations = list(required_type_counts)
    fixed_objects = set(fixed_targets_by_object)
    active_objects = [
        name
        for name, entry in extra.items()
        if isinstance(entry, dict)
        and "pose" in entry
        and "state" in entry
        and name not in locations
    ]

    current_by_cell: Dict[tuple[str, str], List[str]] = {}
    location_by_object: Dict[str, Optional[str]] = {}

    for object_name in active_objects:
        location_by_object[object_name] = None
        for location in locations:
            if location not in extra:
                continue
            if is_object_inside_target(
                extra[object_name]["pose"][env_id],
                extra[location]["pose"][env_id],
                thresh=thresh,
                keep_tensor=False,
            ):
                location_by_object[object_name] = location
                if object_name not in fixed_objects:
                    current_by_cell.setdefault(
                        (location, _object_type(object_name)), []
                    ).append(object_name)
                break

    matching_count = 0
    deficit_types = set()
    surplus_cells = set()

    for location, type_counts in required_type_counts.items():
        fixed_count_by_type: Dict[str, int] = {}
        for object_name, target in fixed_targets_by_object.items():
            if target != location:
                continue
            object_type = _object_type(object_name)
            fixed_count_by_type[object_type] = (
                fixed_count_by_type.get(object_type, 0) + 1
            )

        for object_type, target_count in type_counts.items():
            needed_count = target_count - fixed_count_by_type.get(object_type, 0)
            cell = (location, object_type)
            current_count = len(current_by_cell.get(cell, []))

            if needed_count <= 0:
                if current_count > 0:
                    surplus_cells.add(cell)
                continue

            matching_count += min(current_count, needed_count)

            if current_count < needed_count:
                deficit_types.add(object_type)
            if current_count > needed_count:
                surplus_cells.add(cell)

    fixed_candidates = []
    for object_name, target in fixed_targets_by_object.items():
        if location_by_object.get(object_name) == target:
            matching_count += 1
        elif object_name in location_by_object:
            fixed_candidates.append(object_name)

    if matching_count != minimum - 1:
        return []

    interchangeable_candidates = [
        object_name
        for cell in surplus_cells
        if cell[1] in deficit_types
        for object_name in current_by_cell.get(cell, [])
    ]

    # Objects outside a configured location also provide a direct useful move
    # when their type has a deficient destination.
    interchangeable_candidates.extend(
        object_name
        for object_name, location in location_by_object.items()
        if object_name not in fixed_objects
        and location is None
        and _object_type(object_name) in deficit_types
    )

    candidates = fixed_candidates + interchangeable_candidates
    max_safe = min(max_nb, len(candidates) - 1)
    if max_safe <= 0:
        return []

    return random.sample(candidates, k=random.randint(1, max_safe))


class GraspItemsFailureError(GraspFailureError):
    def __init__(
        self,
        max_impossible: int = 0,
        required_type_counts: Optional[Dict[str, Dict[str, int]]] = None,
        fixed_targets_by_object: Optional[Dict[str, str]] = None,
        minimum: int = 0,
        thresh: float = 0.2,
    ) -> None:
        super().__init__()
        self.max_nb = max_impossible
        self.required_type_counts = (
            copy.deepcopy(required_type_counts)
            if required_type_counts is not None
            else None
        )
        self.fixed_targets_by_object = copy.deepcopy(
            fixed_targets_by_object or {}
        )
        self.minimum = minimum
        self.thresh = thresh

    def _to_spec_arguments(self) -> Dict[str, Any]:
        return {
            "max_impossible": self.max_nb,
            "required_type_counts": copy.deepcopy(self.required_type_counts),
            "fixed_targets_by_object": self.fixed_targets_by_object.copy(),
            "minimum": self.minimum,
            "thresh": self.thresh,
        }

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        if self.required_type_counts is None:
            return {"inaccessible": []}

        return {
            "inaccessible": _sample_error_objects(
                obs,
                env_id,
                self.required_type_counts,
                self.fixed_targets_by_object,
                self.minimum,
                self.max_nb,
                self.thresh,
            )
        }


class MaskItemsError(MaskedObjectError):
    def __init__(
        self,
        max_masking: int = 0,
        required_type_counts: Optional[Dict[str, Dict[str, int]]] = None,
        fixed_targets_by_object: Optional[Dict[str, str]] = None,
        minimum: int = 0,
        thresh: float = 0.2,
    ) -> None:
        super().__init__(tool_execution_target_key="target_name")
        self.max_nb = max_masking
        self.required_type_counts = (
            copy.deepcopy(required_type_counts)
            if required_type_counts is not None
            else None
        )
        self.fixed_targets_by_object = copy.deepcopy(
            fixed_targets_by_object or {}
        )
        self.minimum = minimum
        self.thresh = thresh

    def _to_spec_arguments(self) -> Dict[str, Any]:
        return {
            "max_masking": self.max_nb,
            "required_type_counts": copy.deepcopy(self.required_type_counts),
            "fixed_targets_by_object": self.fixed_targets_by_object.copy(),
            "minimum": self.minimum,
            "thresh": self.thresh,
        }

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        if self.required_type_counts is None:
            return {"masked": []}

        return {
            "masked": _sample_error_objects(
                obs,
                env_id,
                self.required_type_counts,
                self.fixed_targets_by_object,
                self.minimum,
                self.max_nb,
                self.thresh,
            )
        }

    def apply_post_verif(
        self,
        tool_result: ToolResult,
        arguments: Dict[str, Any],
    ) -> None:
        if not tool_result.context:
            return

        masked = set(arguments.get("masked", []))
        if not masked:
            return

        visible_by_location = {
            location: {
                name: value
                for name, value in objects.items()
                if name not in masked
            }
            for location, objects in tool_result.context.items()
        }
        descriptions = []
        for location, objects in visible_by_location.items():
            if objects:
                descriptions.append(
                    f"{location} contains {', '.join(objects)}"
                )
            else:
                descriptions.append(f"{location} is empty")

        tool_result.context = visible_by_location
        tool_result.reason = "Detected objects: " + ". ".join(descriptions) + "."


def _objects_on_table(
    obs: Observation,
    env_id: int,
    active_objects: List[str],
    thresh: float,
) -> List[str]:
    extra = obs.maniskill_obs["extra"]
    if "table" not in extra:
        return []

    return [
        object_name
        for object_name in active_objects
        if object_name in extra
        and is_object_inside_target(
            extra[object_name]["pose"][env_id],
            extra["table"]["pose"][env_id],
            thresh=thresh,
            keep_tensor=False,
        )
    ]


def _sample_clear_table_objects(
    obs: Observation,
    env_id: int,
    active_objects: List[str],
    maximum_remaining: int,
    max_nb: int,
    thresh: float,
) -> List[str]:
    table_objects = _objects_on_table(obs, env_id, active_objects, thresh)
    if len(table_objects) != maximum_remaining + 1:
        return []

    max_safe = min(max_nb, len(table_objects) - 1)
    if max_safe <= 0:
        return []
    return random.sample(table_objects, k=random.randint(1, max_safe))


def _sample_set_table_objects(
    obs: Observation,
    env_id: int,
    active_objects: List[str],
    requirements: Dict[str, int],
    minimum: int,
    max_nb: int,
    thresh: float,
) -> List[str]:
    table_objects = set(_objects_on_table(obs, env_id, active_objects, thresh))
    matching_count = 0
    deficient_requirements = set()

    for requirement, needed_count in requirements.items():
        current_count = sum(
            object_name.rsplit("_", 1)[0] == requirement
            for object_name in table_objects
        )
        matching_count += min(current_count, needed_count)
        if current_count < needed_count:
            deficient_requirements.add(requirement)

    if matching_count != minimum - 1:
        return []

    candidates = [
        object_name
        for object_name in active_objects
        if object_name not in table_objects
        and object_name.rsplit("_", 1)[0] in deficient_requirements
    ]
    max_safe = min(max_nb, len(candidates) - 1)
    if max_safe <= 0:
        return []
    return random.sample(candidates, k=random.randint(1, max_safe))


class MaskClearTableError(MaskItemsError):
    def __init__(
        self,
        active_objects: Optional[List[str]] = None,
        maximum_remaining: int = 0,
        max_masking: int = 0,
        thresh: float = 0.2,
    ) -> None:
        super().__init__()
        self.active_objects = list(active_objects or [])
        self.maximum_remaining = maximum_remaining
        self.max_nb = max_masking
        self.thresh = thresh

    def _to_spec_arguments(self) -> Dict[str, Any]:
        return {
            "active_objects": self.active_objects.copy(),
            "maximum_remaining": self.maximum_remaining,
            "max_masking": self.max_nb,
            "thresh": self.thresh,
        }

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        return {
            "masked": _sample_clear_table_objects(
                obs,
                env_id,
                self.active_objects,
                self.maximum_remaining,
                self.max_nb,
                self.thresh,
            )
        }


class GraspClearTableFailureError(GraspItemsFailureError):
    def __init__(
        self,
        active_objects: Optional[List[str]] = None,
        maximum_remaining: int = 0,
        max_impossible: int = 0,
        thresh: float = 0.2,
    ) -> None:
        super().__init__()
        self.active_objects = list(active_objects or [])
        self.maximum_remaining = maximum_remaining
        self.max_nb = max_impossible
        self.thresh = thresh

    def _to_spec_arguments(self) -> Dict[str, Any]:
        return {
            "active_objects": self.active_objects.copy(),
            "maximum_remaining": self.maximum_remaining,
            "max_impossible": self.max_nb,
            "thresh": self.thresh,
        }

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        return {
            "inaccessible": _sample_clear_table_objects(
                obs,
                env_id,
                self.active_objects,
                self.maximum_remaining,
                self.max_nb,
                self.thresh,
            )
        }


class MaskSetTableError(MaskItemsError):
    def __init__(
        self,
        active_objects: Optional[List[str]] = None,
        requirements: Optional[Dict[str, int]] = None,
        minimum: int = 0,
        max_masking: int = 0,
        thresh: float = 0.2,
    ) -> None:
        super().__init__()
        self.active_objects = list(active_objects or [])
        self.requirements = dict(requirements or {})
        self.minimum = minimum
        self.max_nb = max_masking
        self.thresh = thresh

    def _to_spec_arguments(self) -> Dict[str, Any]:
        return {
            "active_objects": self.active_objects.copy(),
            "requirements": self.requirements.copy(),
            "minimum": self.minimum,
            "max_masking": self.max_nb,
            "thresh": self.thresh,
        }

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        return {
            "masked": _sample_set_table_objects(
                obs,
                env_id,
                self.active_objects,
                self.requirements,
                self.minimum,
                self.max_nb,
                self.thresh,
            )
        }


class GraspSetTableFailureError(GraspItemsFailureError):
    def __init__(
        self,
        active_objects: Optional[List[str]] = None,
        requirements: Optional[Dict[str, int]] = None,
        minimum: int = 0,
        max_impossible: int = 0,
        thresh: float = 0.2,
    ) -> None:
        super().__init__()
        self.active_objects = list(active_objects or [])
        self.requirements = dict(requirements or {})
        self.minimum = minimum
        self.max_nb = max_impossible
        self.thresh = thresh

    def _to_spec_arguments(self) -> Dict[str, Any]:
        return {
            "active_objects": self.active_objects.copy(),
            "requirements": self.requirements.copy(),
            "minimum": self.minimum,
            "max_impossible": self.max_nb,
            "thresh": self.thresh,
        }

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        return {
            "inaccessible": _sample_set_table_objects(
                obs,
                env_id,
                self.active_objects,
                self.requirements,
                self.minimum,
                self.max_nb,
                self.thresh,
            )
        }

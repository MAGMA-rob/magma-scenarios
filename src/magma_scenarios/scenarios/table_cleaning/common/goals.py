from typing import Dict, List, Optional

import torch

from magma_core.simulation.goals import BaseGoal
from magma_core.simulation.utils.env_utils import is_object_inside_target


def _get_pose(entry):
    if isinstance(entry, dict):
        return entry["pose"]
    return entry


def _get_state(entry):
    if isinstance(entry, dict):
        return entry["state"]
    raise ValueError("The observation entry does not expose a state.")


class AtLeastTypedAssignedCount(BaseGoal):
    def __init__(
        self,
        required_type_counts: Dict[str, Dict[str, int]],
        fixed_targets_by_object: Dict[str, str],
        minimum: int,
        thresh: float = 0.2,
    ):
        super().__init__("AtLeastTypedAssignedCount", f"minimum={minimum}")

        self.required_type_counts = required_type_counts
        self.fixed_targets_by_object = fixed_targets_by_object
        self.minimum = minimum
        self.thresh = thresh

    def _object_type(self, obj: str) -> str:
        return obj.rsplit("_", 1)[0]

    def verify(self, obs: Dict) -> torch.Tensor:
        extra = obs["extra"]

        fixed_objects = set(self.fixed_targets_by_object)

        active_objects = [
            name for name, entry in extra.items()
            if isinstance(entry, dict)
            and "pose" in entry
            and "state" in entry
            and name not in fixed_objects
        ]

        reference_pose = _get_pose(
            extra[active_objects[0] if active_objects else next(iter(fixed_objects))]
        )
        nb_envs = 1 if reference_pose.ndim == 1 else reference_pose.shape[0]

        count = torch.zeros(
            nb_envs,
            dtype=torch.int32,
            device=reference_pose.device,
        )

        for location, type_counts in self.required_type_counts.items():
            fixed_objects_in_location = [
                obj
                for obj, target in self.fixed_targets_by_object.items()
                if target == location
            ]

            fixed_count_by_type = {}
            for obj in fixed_objects_in_location:
                obj_type = self._object_type(obj)
                fixed_count_by_type[obj_type] = fixed_count_by_type.get(obj_type, 0) + 1

            for obj_type, needed_count in type_counts.items():
                needed_count -= fixed_count_by_type.get(obj_type, 0)

                if needed_count <= 0:
                    continue

                matching_count = torch.zeros(
                    nb_envs,
                    dtype=torch.int32,
                    device=reference_pose.device,
                )

                for obj in active_objects:
                    if not obj.startswith(obj_type + "_"):
                        continue

                    matching_count += is_object_inside_target(
                        _get_pose(extra[obj]),
                        _get_pose(extra[location]),
                        thresh=self.thresh,
                        keep_tensor=True,
                    ).int()

                count += torch.minimum(
                    matching_count,
                    torch.full_like(matching_count, needed_count),
                )

        for obj, target in self.fixed_targets_by_object.items():
            count += is_object_inside_target(
                _get_pose(extra[obj]),
                _get_pose(extra[target]),
                thresh=self.thresh,
                keep_tensor=True,
            ).int()

        return (count >= self.minimum).int()


class AtLeastTypedStateAssignedCount(BaseGoal):
    def __init__(
        self,
        required_state_counts: Dict[str, Dict[str, Dict[int, int]]],
        minimum: int,
        thresh: float = 0.2,
    ) -> None:
        super().__init__(
            "AtLeastTypedStateAssignedCount",
            f"minimum={minimum}",
        )
        self.required_state_counts = required_state_counts
        self.minimum = minimum
        self.thresh = thresh

    def verify(self, obs: Dict) -> torch.Tensor:
        extra = obs["extra"]
        reference_pose = _get_pose(extra["table"])
        nb_envs = 1 if reference_pose.ndim == 1 else reference_pose.shape[0]
        count = torch.zeros(
            nb_envs,
            dtype=torch.int32,
            device=reference_pose.device,
        )

        active_objects = [
            name
            for name, entry in extra.items()
            if isinstance(entry, dict)
            and "pose" in entry
            and "state" in entry
        ]

        for location, type_counts in self.required_state_counts.items():
            location_pose = _get_pose(extra[location])
            for object_type, state_counts in type_counts.items():
                for expected_state, needed_count in state_counts.items():
                    matching_count = torch.zeros_like(count)
                    for object_name in active_objects:
                        if not object_name.startswith(object_type + "_"):
                            continue
                        state = _get_state(extra[object_name])
                        at_location = is_object_inside_target(
                            _get_pose(extra[object_name]),
                            location_pose,
                            thresh=self.thresh,
                            keep_tensor=True,
                        )
                        matching_count += (
                            at_location & (state == expected_state)
                        ).int()

                    count += torch.minimum(
                        matching_count,
                        torch.full_like(matching_count, needed_count),
                    )

        return (count >= self.minimum).int()


class ObjectsHaveStateGoal(BaseGoal):
    def __init__(self, objects: List[str], expected_state: int, minimum: Optional[int] = None):
        required = len(objects) if minimum is None else minimum

        super().__init__("ObjectsHaveState", f"state={expected_state}, minimum={required}")

        if not objects:
            raise ValueError("objects cannot be empty")

        self.objects = objects
        self.expected_state = expected_state
        self.minimum = required

    def verify(self, obs: Dict) -> torch.Tensor:
        first_state = _get_state(obs["extra"][self.objects[0]])
        count = torch.zeros_like(first_state, dtype=torch.int32)

        for object_name in self.objects:
            state = _get_state(obs["extra"][object_name])
            count += (state == self.expected_state).int()

        return (count >= self.minimum).int()


class AtMostTableObjectCount(BaseGoal):
    def __init__(
        self,
        active_objects: List[str],
        maximum: int,
        thresh: float = 0.2,
    ) -> None:
        super().__init__("AtMostTableObjectCount", f"maximum={maximum}")
        self.active_objects = active_objects.copy()
        self.maximum = maximum
        self.thresh = thresh

    def verify(self, obs: Dict) -> torch.Tensor:
        extra = obs["extra"]
        table_pose = _get_pose(extra["table"])
        nb_envs = 1 if table_pose.ndim == 1 else table_pose.shape[0]
        count = torch.zeros(
            nb_envs,
            dtype=torch.int32,
            device=table_pose.device,
        )

        for object_name in self.active_objects:
            if object_name not in extra:
                continue
            count += is_object_inside_target(
                _get_pose(extra[object_name]),
                table_pose,
                thresh=self.thresh,
                keep_tensor=True,
            ).int()

        return (count <= self.maximum).int()


class AtLeastTableRequirementCount(BaseGoal):
    def __init__(
        self,
        active_objects: List[str],
        requirements: Dict[str, int],
        minimum: int,
        thresh: float = 0.2,
    ) -> None:
        super().__init__("AtLeastTableRequirementCount", f"minimum={minimum}")
        self.active_objects = active_objects.copy()
        self.requirements = requirements.copy()
        self.minimum = minimum
        self.thresh = thresh

    def verify(self, obs: Dict) -> torch.Tensor:
        extra = obs["extra"]
        table_pose = _get_pose(extra["table"])
        nb_envs = 1 if table_pose.ndim == 1 else table_pose.shape[0]
        count = torch.zeros(
            nb_envs,
            dtype=torch.int32,
            device=table_pose.device,
        )

        for requirement, needed_count in self.requirements.items():
            matching_count = torch.zeros_like(count)

            for object_name in self.active_objects:
                if object_name not in extra:
                    continue
                object_type = self._object_type(object_name)
                if object_type != requirement:
                    continue
                matching_count += is_object_inside_target(
                    _get_pose(extra[object_name]),
                    table_pose,
                    thresh=self.thresh,
                    keep_tensor=True,
                ).int()

            count += torch.minimum(
                matching_count,
                torch.full_like(matching_count, needed_count),
            )

        return (count >= self.minimum).int()

    def _object_type(self, object_name: str) -> str:
        return object_name.rsplit("_", 1)[0]

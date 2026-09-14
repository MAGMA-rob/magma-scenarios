import random
from typing import Optional

from magma_core.simulation.constraints import BaseConstraint
from magma_core.simulation.state import TaskState
from magma_scenarios.templates.requests.interact_request import (
    BaseConstraintRequest,
    ConstraintParameters,
)


class ForbidObjectConstraint(BaseConstraint):
    """Persistently mark one object as forbidden for future cycles."""

    def __init__(self, object_name: str) -> None:
        super().__init__()
        self.object_name = object_name

    def apply(self, state: TaskState):
        super().apply(state)
        if self.object_name not in state.attributes.get("objects", []):
            raise RuntimeError(f"The {self.__class__.__name__} failed to be applied")
        state.properties["forbidden_objects"] = [self.object_name]
        state.properties["forbidden_object_needs_application"] = True

    def outdated(self, state: TaskState) -> bool:
        return self.object_name not in state.attributes.get("objects", [])


class AllowObjectConstraint(BaseConstraint):
    """Remove the current object interdiction."""

    def __init__(self, object_name: str) -> None:
        super().__init__()
        self.object_name = object_name

    def apply(self, state: TaskState):
        super().apply(state)
        state.properties["forbidden_objects"] = []
        state.properties.pop("forbidden_object_needs_application", None)

    def outdated(self, state: TaskState) -> bool:
        return self.object_name not in state.attributes.get("objects", [])


class ForbidObjectsRequest(BaseConstraintRequest):
    """Toggle a single persistent object interdiction."""

    def __init__(
            self,
            forbid_sampling_weight: float = 1.0,
            allow_sampling_weight: float = 0.75,
            aged_sampling_weight: float = 4.0,
            requests_to_aged_weight: int = 5,
            steps_to_aged_weight: Optional[int] = None,
        ) -> None:
        super().__init__()
        if forbid_sampling_weight < 0:
            raise ValueError("forbid_sampling_weight must be non-negative")
        if allow_sampling_weight < 0:
            raise ValueError("allow_sampling_weight must be non-negative")
        if aged_sampling_weight < 0:
            raise ValueError("aged_sampling_weight must be non-negative")
        if requests_to_aged_weight <= 0:
            raise ValueError("requests_to_aged_weight must be positive")
        if steps_to_aged_weight is None:
            steps_to_aged_weight = requests_to_aged_weight
        if steps_to_aged_weight <= 0:
            raise ValueError("steps_to_aged_weight must be positive")
        self.forbid_sampling_weight = forbid_sampling_weight
        self.allow_sampling_weight = allow_sampling_weight
        self.aged_sampling_weight = aged_sampling_weight
        self.requests_to_aged_weight = requests_to_aged_weight
        self.steps_to_aged_weight = steps_to_aged_weight

    def _aged_weight(self, state: TaskState, base_weight: float) -> float:
        step_index = state.properties.get("_generator_step_index")
        if step_index is not None:
            last_step_index = state.properties.get("forbid_objects_last_step_index", 0)
            age = max(0, step_index - last_step_index)
            age_to_max_weight = self.steps_to_aged_weight
        else:
            request_index = state.properties.get("_generator_request_index")
            if request_index is None:
                return base_weight
            last_index = state.properties.get("forbid_objects_last_request_index", 0)
            age = max(0, request_index - last_index)
            age_to_max_weight = self.requests_to_aged_weight

        if age <= 0:
            return base_weight

        ratio = min(1.0, age / age_to_max_weight)
        return base_weight + (self.aged_sampling_weight - base_weight) * ratio

    def sampling_weight(self, state: TaskState) -> float:
        all_objects = state.attributes.get("objects", [])
        if len(all_objects) <= 0:
            return 0
        if state.properties.get("forbidden_objects"):
            return self._aged_weight(state, self.allow_sampling_weight)
        if len(all_objects) <= 1:
            return 0
        return self._aged_weight(state, self.forbid_sampling_weight)

    def apply_request(
        self,
        state: TaskState,
        parameters: ConstraintParameters,
    ) -> TaskState:
        state = super().apply_request(state, parameters)
        state.properties["forbid_objects_last_request_index"] = (
            state.properties.get("_generator_request_index", 0)
        )
        state.properties["forbid_objects_last_step_index"] = (
            state.properties.get("_generator_step_index", 0)
        )
        return state

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        forbidden_objects = [
            obj
            for obj in state.properties.get("forbidden_objects", [])
            if obj in state.attributes.get("objects", [])
        ]
        if forbidden_objects:
            object_name = forbidden_objects[0]
            templates = (
                f"{object_name} can be used again now.",
                f"From now on, {object_name} is available again for sorting cycles.",
                f"The maintenance is done: you can manipulate {object_name} again.",
            )
            return ConstraintParameters(
                [AllowObjectConstraint(object_name)],
                random.choice(templates),
            )

        available_objects = [
            obj
            for obj in state.attributes.get("objects", [])
        ]
        if len(available_objects) <= 1:
            raise RuntimeError(
                f"Failed to build the stage from {self.__class__.__name__} "
                "due to too few available objects"
            )

        object_name = random.choice(available_objects)
        templates = (
            f"Please remember that {object_name} must not be manipulated for now.",
            f"New safety rule: do not sort {object_name} until I say otherwise.",
            f"From now on, {object_name} cannot be used in sorting cycles.",
        )
        return ConstraintParameters(
            [ForbidObjectConstraint(object_name)],
            random.choice(templates),
        )

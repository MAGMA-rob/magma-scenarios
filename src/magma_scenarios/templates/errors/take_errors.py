import random
from typing import Any, Dict, Optional

from magma_core.simulation.data_structures.observation import Observation
from magma_core.simulation.data_structures.tools import ToolResult, ToolExecution
from magma_core.simulation.errors import BaseError
from magma_core.simulation.utils.env_utils import is_object_inside_target

class GraspFailureError(BaseError):
    """
    Allow to inject a grasp failure error to the stage.

    It uses the 'inaccessible' keys from arguments. Please overridde the initialize to set this key.
    """

    recovery_extra_steps = 1
    required_key = ["inaccessible"]

    def __init__(self, tool_execution_target_key : str = "target_name") -> None:
        super().__init__()
        self.target_key = tool_execution_target_key

    def initialize(self, obs : Observation, env_id : int) -> Dict[str, Any]:
        raise NotImplementedError()

    def apply_pre_exec(
        self,
        tool_execution: ToolExecution,
        arguments: Dict[str, Any],
    ) -> bool:
        inaccessible = arguments.get("inaccessible",None)
        if inaccessible is None or len(inaccessible)==0:
            return False
        target_name = tool_execution.context.get(self.target_key, None)
        if target_name is None:
            return False
        
        if target_name in inaccessible:
            tool_execution.fail(
                f"Failed to grasp: {target_name}. The object is unreachable right now."
            )
            return True
        return False

    def get_description(self, arguments: Optional[Dict[str, Any]]) -> str:
        if arguments is None or arguments.get("inaccessible") is None or len(arguments["inaccessible"]) == 0:
            return "Make some object impossible to take"
        return f"These objects are impossible to take right now: {arguments['inaccessible']}. Try to grasp another objects that also allows to complete the instruction."


class OneShotToolFailureError(BaseError):
    """Fail one compatible tool call probabilistically before its execution."""

    recovery_extra_steps = 1
    required_key = ["remaining_failures"]

    def __init__(
        self,
        failure_probability: float = 0.2,
        failure_count: int = 1,
    ) -> None:
        if not 0.0 <= failure_probability <= 1.0:
            raise ValueError("failure_probability must be between 0 and 1")
        if failure_count < 0:
            raise ValueError("failure_count must be greater than or equal to 0")
        self.failure_probability = failure_probability
        self.failure_count = failure_count

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        return {"remaining_failures": self.failure_count}

    def apply_pre_exec(
        self,
        tool_execution: ToolExecution,
        arguments: Dict[str, Any],
    ) -> bool:
        remaining_failures = arguments.get("remaining_failures", 0)
        if remaining_failures <= 0:
            return False
        if random.random() >= self.failure_probability:
            return False

        arguments["remaining_failures"] = remaining_failures - 1
        tool_execution.fail(
            "The requested action failed due to a temporary execution error. "
            "Please retry."
        )
        return True

    def get_description(self, arguments: Optional[Dict[str, Any]]) -> str:
        remaining_failures = (
            self.failure_count
            if arguments is None
            else arguments.get("remaining_failures", 0)
        )
        return (
            "A compatible action can fail temporarily before execution. "
            f"Remaining failures: {remaining_failures}."
        )


class RequestedObjectGraspFailureError(GraspFailureError):
    """Make surplus instances of requested object types impossible to grasp."""

    def __init__(
        self,
        required_by_prefix: Optional[Dict[str, int]] = None,
        interchangeable_groups: Optional[list[list[str]]] = None,
        source_locations: Optional[list[str]] = None,
        thresh: float = 0.3,
    ) -> None:
        super().__init__()
        self.required_by_prefix = dict(required_by_prefix or {})
        self.interchangeable_groups = [
            list(group) for group in (interchangeable_groups or [])
        ]
        self.source_locations = list(source_locations or [])
        self.thresh = thresh

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        extra = obs.maniskill_obs["extra"]
        inaccessible = []

        candidate_groups = []
        for prefix, minimum_available in self.required_by_prefix.items():
            candidates = []
            for object_name, entry in extra.items():
                if not object_name.startswith(prefix):
                    continue

                if self.source_locations:
                    object_pose = entry["pose"] if isinstance(entry, dict) else entry
                    if not any(
                        location in extra
                        and is_object_inside_target(
                            object_pose[env_id],
                            (
                                extra[location]["pose"][env_id]
                                if isinstance(extra[location], dict)
                                else extra[location][env_id]
                            ),
                            thresh=self.thresh,
                            keep_tensor=False,
                        )
                        for location in self.source_locations
                    ):
                        continue

                candidates.append(object_name)

            candidate_groups.append((candidates, minimum_available))

        for group in self.interchangeable_groups:
            candidate_groups.append(
                ([name for name in group if name in extra], 1)
            )

        for candidates, minimum_available in candidate_groups:
            candidates.sort()
            blocked_count = len(candidates) - minimum_available
            if blocked_count <= 0:
                continue

            # The first named instance is deliberately blocked because agents
            # commonly try objects in their observed or lexical order.
            blocked = [candidates[0]]
            if blocked_count > 1:
                blocked.extend(
                    random.sample(candidates[1:], k=blocked_count - 1)
                )
            inaccessible.extend(blocked)

        return {"inaccessible": inaccessible}

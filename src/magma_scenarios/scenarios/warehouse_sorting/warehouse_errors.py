from typing import Any, Dict, Optional

from magma_core.base.data_structures import Observation, ToolExecution
from magma_core.base.errors import BaseError


class LaunchCycleTransientFailureError(BaseError):
    """Inject one transient launch_cycle failure, then let retries succeed."""

    recovery_extra_steps = 1
    optional_key = ["message"]

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        return {}

    def apply_pre_exec(self, tool_execution: ToolExecution, arguments: Dict[str, Any]):
        if arguments.get("_triggered", False):
            return

        arguments["_triggered"] = True
        tool_execution.fail(
            arguments.get(
                "message",
                "Communication error with the robot. Please retry the cycle.",
            )
        )

    def get_description(self, arguments: Optional[Dict[str, Any]]) -> str:
        return "The first launch_cycle call fails once with a transient communication error."

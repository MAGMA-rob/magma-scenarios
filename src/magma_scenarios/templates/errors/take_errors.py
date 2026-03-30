from typing import Dict, Any

from magma_core.base.data_structures.observation import Observation
from magma_core.base.data_structures.tools import ToolResult, ToolExecution
from magma_core.base.errors import BaseError

class GraspFailureError(BaseError):
    """
    Allow to inject a grasp failure error to the stage.

    It uses the 'innaccessible' keys from arguments. Please overridde the initialize to set this key.
    """

    recovery_extra_steps = 1

    def __init__(self, tool_execution_target_key : str = "target_name") -> None:
        super().__init__()
        self.target_key = tool_execution_target_key

    def initialize(self, obs : Observation, env_id : int) -> Dict[str, Any]:
        raise NotImplementedError()

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
        if arguments is None or arguments.get("innaccessible") is None or len(arguments["innaccessible"] == 0):
            return "Make some object impossible to take"
        return f"These objects are impossible to take right now: {arguments['innaccessible']}. Try to grasp another objects that also allows to complete the instruction."
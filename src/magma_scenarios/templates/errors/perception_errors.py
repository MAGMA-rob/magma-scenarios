from typing import Dict, Any

from magma_core.base.data_structures.observation import Observation
from magma_core.base.data_structures.tools import ToolResult, ToolExecution
from magma_core.base.errors import BaseError
 

class MaskedObjectError(BaseError):
    """
    Delete object from the observation returned to the agent. It works with tools
    that return a dict of 'area':list_of_object or a list_of_object.

    You must inherit from it and define the apply_post_verif and initialize
    """

    recovery_extra_steps = 0

    def __init__(self, tool_execution_target_key : str = "target_name") -> None:
        super().__init__()
        self.target_key = tool_execution_target_key

    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any] | None:
        raise NotImplementedError("This class msut be defined in child class")

    def apply_pre_exec(self, tool_execution: ToolExecution, arguments: Dict[str, Any]):
        """
        Allows to ensure that masked object are not taken or manipulated.
        """
        masked = arguments.get("masked",None)
        if masked is None or len(masked)==0:
            return
        target_name = tool_execution.context.get(self.target_key, None)
        if target_name is None:
            return

        if target_name in masked:
            tool_execution.fail(
                f"Unknown object: {target_name}. Please use only detected objects."
            )

    def apply_post_verif(self, tool_result: ToolResult, arguments: Dict[str, Any]):
        raise NotImplementedError("This function must be defined in the child class to allow custom modification")

    def get_description(self, arguments: Dict[str, Any] | None) -> str:
        if arguments is None or arguments.get("masked") is None or len(arguments['masked']) == 0:
            return ""
        return f"Masked objects from perception: {arguments['masked']}. One solution is to take another objects that still resolves the instruction."
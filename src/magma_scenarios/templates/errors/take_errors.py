

from magma_core.base.data_structures.tools import ToolResult
from magma_core.base.errors import BaseError

class ImpossibleGraspError(BaseError):
    """
    Make an object impossible to grasp
    """

    recovery_extra_steps = 1

    def __init__(self) -> None:
        super().__init__()

    # def apply(self, tool_result: ToolResult):
    #     if not tool_result.context or len(tool_result.context) == 0:
    #         raise RuntimeError("The Localization Error was activated on a tool that does not return the context dict")

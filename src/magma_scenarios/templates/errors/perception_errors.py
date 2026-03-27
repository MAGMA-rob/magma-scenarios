

from magma_core.base.data_structures.tools import ToolResult
from magma_core.base.errors import BaseError

# class BadLocalizationError(BaseError):
#     """
#     This error represent errors in object localization. It typically work with tools
#     that return a dict of 'area':list_of_object.

#     It will modify the position of objects to another areas.
#     """
#     recovery_extra_steps = 1

#     def __init__(self) -> None:
#         super().__init__()


#     def apply(self, tool_result: ToolResult):
#         if not tool_result.context or len(tool_result.context) == 0:
#             raise RuntimeError("The Localization Error was activated on a tool that does not return the context dict")
        

class MaskedObjectError(BaseError):
    """
    Delete object from the observation returned to the agent. It works with tools
    that return a dict of 'area':list_of_object or a list_of_object.
    """

    recovery_extra_steps = 1

    def __init__(self) -> None:
        super().__init__()

    def apply(self, tool_result: ToolResult):
        if not tool_result.context or len(tool_result.context) == 0:
            raise RuntimeError("The Localization Error was activated on a tool that does not return the context dict")

from magma_core.base.data_structures.tools import ToolErrorSupport
from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.base.data_structures import ToolExecution, ToolResult, Observation
from magma_scenarios.scenarios.color_sorting.color_sorting_errors import MaskRemainingCubesError
from .cooking_errors import MaskRemainingFoodError, GraspFoodFailureError


class CookingTool(BaseToolsAPI):

    @register_tool(
        description ="Returns visible objects and their locations",
        params_spec={},
        errors = [
            ToolErrorSupport(MaskRemainingFoodError,pre = True, post= False)
        ]
    )
    def detect(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        pass


    @register_tool(
        description ="Pick an object from the table",
        params_spec={"name" : {"description" : "the name of object to take", "type": str}},
        errors = [
            ToolErrorSupport(GraspFoodFailureError,pre = True, post= False)
        ]
    )
    def take(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        pass

    @register_tool(
        description ="Place the held object either on the table or on the plate",
        params_spec={"target" : {"description" : "the name of the target where put the object", "type" : str}}
    )
    def put(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        pass

    @register_tool(
        description ="Used to validate whether constraints are satisfied",
        params_spec={}
    )
    def valid_plate(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        pass
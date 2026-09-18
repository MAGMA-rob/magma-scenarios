from typing import Dict
from magma_core.simulation.data_structures import (
    Log,
    Observation,
    ToolErrorSupport,
    ToolExecution,
    ToolResult,
)
from magma_core.simulation.tools import BaseToolsAPI, register_tool
from magma_core.simulation.utils.gripper_utils import is_object_in_gripper, find_object_in_gripper
from magma_scenarios.utils import compute_swipe_trajectory
from ..common.attributes import DISHWASHER_LOCATIONS, cleaning_objects, dishware, food
from ..common.tool_helpers import TableCleaningToolHelpers
from ..common.errors import GraspItemsFailureError, MaskItemsError


class TableCleaningTool(BaseToolsAPI, TableCleaningToolHelpers):
    r = 0.11
    table_gride_centre = [-0.1,-0.2,0]
    machine_center = [-0.8, -0.5, 0.02]
    trash_centre = [-1,0.5,0]
    food_storage_centre = [-0.25, 0.25,0]
    dish_storage_centre = [-0.05, 0.25,0]

    @register_tool(
        description ="Returns visible objects and their locations",
        params_spec={},
        is_detection=True,
        errors=[
            ToolErrorSupport(MaskItemsError, pre=False, post=True),
        ],
    )
    def detect(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        
        extra = obs.maniskill_obs["extra"]
        detected_obj = {"table" : {},"trashcan" : {},"washing_machine" : {},"food_storage" : {},"dish_storage" : {}}
        for name , data in extra.items() :
            if name in ["agent_tcp","washing_machine","food_storage","dish_storage","trashcan","table"]:
                continue

            pos = data["pose"][env_id][:3]
            x, y, _ = pos

            if x> -0.4 and y < 0 :
                detected_obj["table"][name] = pos
            if x < -0.6 and y > 0.2 :
                detected_obj["trashcan"][name] = pos
            if x < -0.4 and y < 0 :
                detected_obj["washing_machine"][name] = pos
            if -0.4< x <-0.12 and y > 0 :
                detected_obj["food_storage"][name] = pos
            if x > -0.12 and y > 0 :
                detected_obj["dish_storage"][name] = pos


        def verifier(new_obs: Dict) -> ToolResult:
            if all(not v for v in detected_obj.values()) : 
                return ToolResult(True,"there is no object detected",context = detected_obj)
            s = "this is the position of existing object :"

            for location, objets in detected_obj.items() :
                if len(objets) == 0 :
                    continue
                objs = ', '.join(f"{obj}" for obj in objets.keys())
                s += f" {location} contain {objs}."
                

            return ToolResult(True,s,context = detected_obj)

        return ToolExecution(poses = ["OK"], verifier=verifier)
        

    def is_table_empty(self,table_gride_centre,obs,env_id,thresh) :
        thresh = thresh 
        extra = obs.maniskill_obs["extra"]
        for obj, data in extra.items() :
            if obj == "table" :
                continue
            pose = data["pose"][env_id][:3]
            x, y, z = pose
            if (table_gride_centre[0]-thresh < x < table_gride_centre[0]+thresh) and (table_gride_centre[1]-thresh < y < table_gride_centre[1]+thresh) and z < 0.03:
                return False
        return True
    
    @register_tool(
        description ="Clean the table using the sponge",
        params_spec={}
    )    
    def wipe(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        extra = obs.maniskill_obs["extra"]
        reduced_env = dict((k, v["pose"][env_id][:3]) for k, v in extra.items())
        agent_tcp_position = reduced_env.pop("agent_tcp")
        for obj, pos in reduced_env.items() :
            if obj == "sponge" : 
                continue
            obj_in_gripper = is_object_in_gripper(agent_tcp_position, pos, threshold=0.02)
            if obj_in_gripper:
                return ToolExecution(
                        poses = [], verifier= None, reason=f"Cannot clean while holding {obj}"
                    )
        obj_in_gripper = is_object_in_gripper(
            agent_tcp_position,
            extra["sponge"]["pose"][env_id],
            threshold=0.02,
        )
        if not obj_in_gripper:
            return ToolExecution(
                poses = [], verifier= None, reason="You must take a cleaning object to clean the table.")

        thresh = 0.12
        if not self.is_table_empty(self.table_gride_centre,obs,env_id,thresh) :
            return ToolExecution(
                poses = [], verifier= None, reason="The table is not empty."
            )
        poses = compute_swipe_trajectory(self.get_agent(),self.table_gride_centre,thresh,0.01)

        def verifier(new_obs: dict) -> ToolResult:
            #the sponge must be always in the gripper
            new_extra = obs.maniskill_obs["extra"]
            obj_in_gripper = is_object_in_gripper(
                agent_tcp_position,
                new_extra["sponge"]["pose"][env_id],
                threshold=0.02,
            )
            if not obj_in_gripper:
                return ToolResult(False, reason="You dropped the sponge during the cleaning action")
            return ToolResult(True,reason="you cleaned the table successfully",logs=Log(""))

        return ToolExecution(
            poses,
            verifier=verifier,
            allowed_moving_actors=["sponge"],
        )

    @register_tool(
        description="Pick up an object",
        params_spec={"name": {"description": "The name of the object to take","type": str,}},
        errors=[
            ToolErrorSupport(MaskItemsError, pre=True, post=False),
            ToolErrorSupport(GraspItemsFailureError, pre=True, post=False),
        ],
    )
    def take(self, obs: Observation, env_id: int, params: dict) -> ToolExecution:
        return self._take_object( obs=obs, env_id=env_id, object_name=params["name"], tcp_key="agent_tcp")

    @register_tool(
        description="Place the held object on a target location",
        params_spec={"target": {"description": "Target location","type": str}},
    )
    def put(self, obs: Observation, env_id: int, params: dict) -> ToolExecution:
        target = params["target"]

        if target not in DISHWASHER_LOCATIONS:
            return self._return_failed_tool( f"Unknown target: {target}.")

        return self._put_object(obs=obs, env_id=env_id, target=target, tcp_key="agent_tcp", container_targets=("trashcan", "washing_machine"))

    @register_tool(
        description="Adjust an ambient setting without affecting the workspace.",
        params_spec={
            "state": {
                "description": "Requested ambient setting.",
                "type": str,
            }
        },
    )
    def set_light(self, obs: Observation, env_id: int, params: dict) -> ToolExecution:
        def verifier(new_obs: Dict) -> ToolResult:
            return ToolResult(True, reason="The ambient setting was acknowledged.")

        return ToolExecution(poses=["OK"], verifier=verifier)

    def _plural(self,objects) :
        if len(objects) == 1 :
            return "is"
        return "are"

    @register_tool(
        description ="Inspect the object to determine if it is dirty or clean.",
        params_spec={},
        is_detection=True,
    )
    def inspect(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        extra = obs.maniskill_obs["extra"]
        reduced_env = {obj : data["pose"][env_id][:3] for obj, data in extra.items() if obj in [*food, *dishware, *cleaning_objects]}

        agent_tcp_position = extra["agent_tcp"]["pose"][env_id][:3]
        obj_in_gripper = find_object_in_gripper(agent_tcp_position, reduced_env)

        if not obj_in_gripper:
            return ToolExecution(poses=[],verifier=None,reason="No object in the gripper. Inspection failed")
        
        state = extra[obj_in_gripper]["state"][env_id]

        def verifier(new_obs: Dict) -> ToolResult:
            if state ==1:
                s = f"Object {obj_in_gripper} is clean"
            else:
                s = f"Object {obj_in_gripper} is dirty"
            return ToolResult(True,s)

        return ToolExecution(poses = ["OK"], verifier=verifier)

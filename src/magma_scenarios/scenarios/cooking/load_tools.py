from typing import Dict
from magma_core.base.data_structures.tools import ToolErrorSupport
from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.base.data_structures import ToolExecution, ToolResult, Observation, Log
from magma_core.utils.gripper_utils import find_object_in_gripper, is_object_in_gripper
from magma_scenarios.envs.six_cubes_two_boxes_on_table import reduced_env
import sapien
from .cooking_errors import MaskRemainingFoodError, GraspFoodFailureError
from magma_scenarios.utils import compute_drop_trajectory, compute_grasp_trajectory
from magma_core.utils.env_utils import is_object_inside_target
import numpy as np
from .attributes import fruits, drinks, main_course

class CookingTool(BaseToolsAPI):

    r = 0.17
    table_gride_centre = [0.02, -0.2 ,0]


    def _world_to_grid(self,center_target_position : list , object_world_position : list, r : float)-> tuple:
        i = np.round((object_world_position[0]-center_target_position[0])/r)
        j = np.round((object_world_position[1]-center_target_position[1])/r)        
        return (i,j)

    def _grid_to_world(self,grid : tuple , center_target_position : list, r : float):
        grid_position = center_target_position.copy()
        grid_position[0] = grid_position[0] + r * grid[0]
        grid_position[1] = grid_position[1] + r * grid[1]
        return grid_position

    def _get_free_cell(self,obs : Observation , env_id : int ,target_center : list):
        extra = obs.maniskill_obs["extra"]
        gride = [(1,1),(0,1),(-1,1),
                 (1,0),(0,0),(-1,0),
                (1,-1),(0,-1),(-1,-1)]
        occupide = []
        for name, pos in extra.items() :
            if name in ["agent_tcp", "tray"] :
                continue
            object_position = pos[env_id].cpu().numpy()
            if np.allclose(target_center ,self.table_gride_centre) and object_position[1] > 0 :
                continue
            if not np.allclose(target_center ,self.table_gride_centre) and object_position[1] < 0 :
                continue
            grid_object_position = self._world_to_grid(target_center,object_position,self.r)
            occupide.append(grid_object_position)
        for grid_position in gride :
            if grid_position not in occupide :
                return grid_position
        return None



    @register_tool(
        description ="Returns visible objects and their locations",
        params_spec={},
        errors = [
            ToolErrorSupport(MaskRemainingFoodError,pre = True, post= False)
        ]
    )
    def detect(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        
        extra = obs.maniskill_obs["extra"]
        detected_obj = {"table" : {},"tray" : {}}
        for name , pos in extra.items() :
            if name in ["agent_tcp","tray"]:
                continue
            y = pos[env_id][1]
            if y < 0 :
                detected_obj["table"][name] = pos[env_id][:3].cpu().numpy()
            else :
                detected_obj["tray"][name] = pos[env_id][:3].cpu().numpy()  
        def verifier(new_obs: Dict) -> ToolResult:
            s = "this is the position of existion object : "

            if len(detected_obj["table"]) == 0 :
                s += "table is empty"
            else :
                s += "table contain " + ', '.join(f"{name}" for name in detected_obj["table"].keys())

            if len(detected_obj["tray"]) == 0 :
                s += "and tray is empty"
            else :
                s += "and tray contain " + ', '.join(f"{name}" for name in detected_obj["tray"].keys())
            

            return ToolResult(True,s,context = detected_obj, logs = Log(""))

        return ToolExecution(poses = ["OK"], verifier=verifier)


    @register_tool(
        description ="Pick an object from the table",
        params_spec={"name" : {"description" : "the name of object to take", "type": str}},
        errors = [
            ToolErrorSupport(GraspFoodFailureError,pre = True, post= False)
        ]
    )
    def take(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        """go fetch an object by his name """
        extra = obs.maniskill_obs["extra"]
        name = params["name"]
        object = extra.get(name, None)
        if object is None:
            return ToolExecution(
                poses = [], verifier=None, reason=f"no object with name {params['name']}"
            )
        poses = compute_grasp_trajectory(self.get_agent(),object["pose"][env_id].cpu().numpy())

        def verifier(new_obs: dict) -> ToolResult:
            "the object must be in the gripper"
            new_extra = new_obs["extra"]
            object = new_extra.get(name, None)
            if object is None:
                return ToolResult(False, f"the object {name} does not exist anymore ")
            if is_object_in_gripper(new_extra["agent_tcp"]["pose"][env_id],
                                    object["pose"][env_id],
                                    threshold=0.005) :
                return ToolResult(True, f"you have {name} in your gripper")
            else :
                return ToolResult(
                    False, f"you failed to take the object {name}. you can retry"
                )
        return ToolExecution(poses, verifier=verifier, context = {"target_name": name})


    @register_tool(
        description ="Place the held object either on the table or on the tray",
        params_spec={"target" : {"description" : "the name of the target where put the object", "type" : str}}
    )
    def put(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        extra = obs.maniskill_obs["extra"]
        reduced_env = {obj : pose[env_id][:3] for obj, pose in extra.items()}
        agent_tcp_position = reduced_env.pop("agent_tcp")
        obj_in_gripper = find_object_in_gripper(agent_tcp_position, reduced_env)

        if obj_in_gripper is None:
            return ToolExecution(
                poses = [], verifier= None, reason = "No food in gripper"
            )
     
        if params["target"] == "table" :
            target_center = self.table_gride_centre
        elif params["target"] == "tray" :
            target_center = extra["tray"]["pose"][env_id][:3].cpu().numpy()
        target_cell = self._get_free_cell(obs,env_id,target_center)
        if target_cell is None :
            r = f"there is no place in the {params["target"]}"
            return ToolExecution(
                poses = [], verifier= None, reason = r
            )
        target_pos = self._grid_to_world(target_cell, target_center, self.r)

        poses = [sapien.Pose(agent_tcp_position[:3].cpu().numpy() + (0, 0, 0.1), (0, 1, 0, 0))]
        poses.extend(compute_drop_trajectory(
            self.get_agent(),
            drop_pose = target_pos,
            drop_seuil = 0.2,
            approach_seuil = 0.3
        ))

        def verifier(new_obs: dict) -> ToolResult:
            new_extra = new_obs["extra"]
            obj_pose = new_extra[obj_in_gripper]["pose"][env_id]
            if is_object_in_gripper(
                new_extra["agent_tcp"]["pose"][env_id], obj_pose):
                return ToolResult(False, reason = "the food is still in the gripper")
            if not is_object_inside_target(obj_pose, target_pos) :
                return ToolResult(False, reason=f"The {obj_in_gripper} is not in {params['target']} and not in the gripper")
            return ToolResult(True,reason=f"Successfully put {obj_in_gripper} in the {params['target']}")

        return ToolExecution(poses, verifier)
            

    @register_tool(
        description ="Used to validate whether constraints are satisfied",
        params_spec={}
    )
    def valid_plate(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        "the plate must contain at least one food"
        "the plate must not have two food from the category"
        def verifier(new_obs: dict)-> ToolResult:
            extra = new_obs["extra"]
            tray_content = {"fruits" : [], "drinks": [], "main_course": []}
            for name, pose in extra.items() :
                if name in ["agent_tcp", "tray"] :
                    continue
                if is_object_inside_target(pose[env_id], extra["tray"]["pose"][env_id]) :
                    if name in fruits :
                        tray_content["fruits"].append(name)
                    if name in drinks :
                        tray_content["drinks"].append(name)
                    if name in main_course :
                        tray_content["main_course"].append(name)
            if all(len(v) == 0 for v in tray_content.values()) :
                return ToolResult(False,reason=f"there is no food in the tray")

            if len(tray_content["drinks"]) > 1 or len(tray_content["fruits"]) > 1 or len(tray_content["main_course"]) > 1 :
                return ToolResult(False,f"there is more than one food from the same category")

            return ToolResult(True,f"the tray is ready ! ",logs = Log(content=tray_content))
        return ToolExecution(["OK"], verifier=verifier)
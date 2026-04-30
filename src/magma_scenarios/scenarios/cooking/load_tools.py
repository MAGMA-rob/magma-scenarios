from typing import Dict
from magma_core.base.data_structures.tools import ToolErrorSupport
from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.base.data_structures import ToolExecution, ToolResult, Observation, Log
from magma_core.utils.gripper_utils import find_object_in_gripper, is_object_in_gripper
from magma_scenarios.envs.six_cubes_two_boxes_on_table import reduced_env
import sapien
from .cooking_errors import MaskFoodError, GraspFoodFailureError
from magma_scenarios.utils import compute_drop_trajectory, compute_grasp_trajectory
from magma_core.utils.env_utils import is_object_inside_target
import numpy as np
import torch
from .attributes import fruits, drinks, main_course

class CookingTool(BaseToolsAPI):

    r = 0.11
    table_gride_centre = [-0.1,-0.2,0]
    tray_gride_center = [-0.1, 0.12,0]



    def _world_to_grid(self,center_target_position : list , object_world_position : list)-> tuple:
        i = np.round((object_world_position[0]-center_target_position[0])/self.r)
        j = np.round((object_world_position[1]-center_target_position[1])/self.r)        
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
            if np.allclose(target_center ,self.tray_gride_center) and object_position[1] < 0 :
                continue
            grid_object_position = self._world_to_grid(target_center,object_position)
            occupide.append(grid_object_position)
        for grid_position in gride :
            if grid_position not in occupide :
                print(grid_position)
                return grid_position
        return None



    @register_tool(
        description ="Returns visible objects and their locations",
        params_spec={},
        errors = [
            ToolErrorSupport(MaskFoodError,pre = True, post= False)
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
            s = "this is the position of existing object : "

            if len(detected_obj["table"]) == 0 :
                s += "the table is empty"
            else :
                s += "the table contain " + ', '.join(f"{name}" for name in detected_obj["table"].keys())

            if len(detected_obj["tray"]) == 0 :
                s += ", and the tray is empty"
            else :
                s += ", and the tray contain " + ', '.join(f"{name}" for name in detected_obj["tray"].keys())
            

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
        poses = compute_grasp_trajectory(self.get_agent(),object[env_id].cpu().numpy())
        all_food = [name for name in extra if name not in ["agent_tcp", "tray"]]
        def verifier(new_obs: dict) -> ToolResult:
            "the object must be in the gripper"
            new_extra = new_obs["extra"]
            object = new_extra.get(name, None)
            if object is None:
                return ToolResult(False, f"the object {name} does not exist anymore ")
            if is_object_in_gripper(new_extra["agent_tcp"][env_id],
                                    object[env_id],
                                    threshold=0.005) :
                return ToolResult(True, f"you have {name} in your gripper")
            else :
                return ToolResult(
                    False, f"you failed to take the object {name}. you can retry"
                )
        return ToolExecution(poses, verifier=verifier, context = {"visible_food": all_food,"target_name": name})


    @register_tool(
        description ="Place the held object either on the table or on the tray",
        params_spec={"target" : {"description" : "the name of the target where put the object", "type" : str}}
    )
    def put(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        extra = obs.maniskill_obs["extra"]
        reduced_env = {obj : pose[env_id][:3] for obj, pose in extra.items()}
        agent_tcp_position = reduced_env.pop("agent_tcp")
        obj_in_gripper = find_object_in_gripper(agent_tcp_position, reduced_env)
        print("Objects:", reduced_env)
        print("TCP:", agent_tcp_position)
        print("Detected in gripper:", obj_in_gripper)
        if obj_in_gripper is None:
            return ToolExecution(
                poses = [], verifier= None, reason = "No food in gripper"
            )

        if params["target"] == "table" :
            target_center = self.table_gride_centre
        elif params["target"] == "tray" :
            target_center = self.tray_gride_center

        target_cell = self._get_free_cell(obs,env_id,target_center)
        if target_cell is None :
            r = f"there is no place in the {params["target"]}"
            return ToolExecution(
                poses = [], verifier= None, reason = r
            )
        target_pose = self._grid_to_world(target_cell, target_center, self.r)

        poses = [sapien.Pose(agent_tcp_position[:3].cpu().numpy() + (0, 0, 0.1), (0, 1, 0, 0))]
        poses.extend(compute_drop_trajectory(
            self.get_agent(),
            drop_pose = target_pose,
            drop_seuil = 0.03,
            approach_seuil = 0.1
        ))

        def verifier(new_obs: dict) -> ToolResult:
            new_extra = new_obs["extra"]
            obj_pose = new_extra[obj_in_gripper][env_id]
            if is_object_in_gripper(
                new_extra["agent_tcp"][env_id], obj_pose):
                return ToolResult(False, reason = "the food is still in the gripper")
            tensor_target_pose = torch.tensor(target_pose, device=obj_pose.device)
            if not is_object_inside_target(obj_pose, tensor_target_pose) :
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
                if is_object_inside_target(pose[env_id], extra["tray"][env_id],thresh=0.2) :
                    if name in fruits :
                        tray_content["fruits"].append(name)
                    if name in drinks :
                        tray_content["drinks"].append(name)
                    if name in main_course :
                        tray_content["main_course"].append(name)

            return ToolResult(True,f"the tray is ready ! ",logs = Log(content=tray_content))
        return ToolExecution(["OK"], verifier=verifier)
# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.base.data_structures import ToolExecution, ToolResult, Observation
from magma_core.utils.env_utils import is_object_inside_target
from magma_scenarios.utils import compute_grasp_drop_trajectory

from typing import Dict, List
import sapien

class BiRobotTools(BaseToolsAPI):

    @register_tool( 
        description="Return the list of existing objects and their position.",
        params_spec={}
    )
    def get_near_objects(self, obs : Observation, env_id : int, params: Dict) -> ToolExecution:

        def verifier(new_obs: Dict) -> ToolResult:
            # Check if the object is no longer in the gripper and is now in the box
            zones = {'mutual':[],'right':[],'left':[]}
            for obj_name, obj_pose in new_obs['extra'].items():
                if "obj" in obj_name:
                    if is_object_inside_target(obj_pose[env_id], obs.maniskill_obs['extra']["right_zone"][env_id]):
                        zones['right'].append(obj_name)
                    elif is_object_inside_target(obj_pose[env_id], obs.maniskill_obs['extra']["left_zone"][env_id]):
                        zones['left'].append(obj_name)
                    elif is_object_inside_target(obj_pose[env_id], obs.maniskill_obs['extra']["mutual_zone"][env_id]):
                        zones['mutual'].append(obj_name)
                    elif obj_pose[env_id][1] > 0:
                        zones['right'].append(obj_name)
                    else:
                        zones["left"].append(obj_name)

            s = "This is the position of existing objects: "

            if len(zones['right']) != 0:
                s += ",".join(zones["right"]) + " are in the right area. "

            if len(zones['left']) != 0:
                s += ",".join(zones['left']) + " are in the left area. "

            if len(zones['mutual']) > 0:
                s += ",".join(zones['mutual']) + " are in the mutual zone."

            return ToolResult(True,s)

        return ToolExecution(poses=["OK"],verifier=verifier,reason="")
    
    @register_tool(
        description="Make the robot take the object and send it to area. The area and the object must be in the range of the robot or it will raise an error.",
        params_spec={
                "object": {"description": "The name of the object to take", "type":str},
                "target_area": {"description": "The target area where to depose the object.", "type":str}
            }
    )
    def deplace(self, obs : Observation, env_id : int, params: Dict) -> ToolExecution:        
        robot_name = obs.selected_robot_name
        object_name = params['object']
        target_area = params['target_area']

        if not target_area in obs.task_attributes["known_area"]:
            return BaseToolsAPI._return_failed_tool(f"You used an unknow area : {target_area}.")
        elif not object_name in obs.maniskill_obs['extra']:
            return BaseToolsAPI._return_failed_tool(f"You used an unknow object name : {object_name}.")

        if robot_name == "arm1" and target_area == "right":
            return BaseToolsAPI._return_failed_tool(f"The area {target_area} is out of range for robot {robot_name}.")
        elif robot_name == "arm2" and target_area == "left":
            return BaseToolsAPI._return_failed_tool(f"The area {target_area} is out of range for robot {robot_name}.")


        obj_pos = obs.maniskill_obs['extra'][object_name][env_id]
        if is_object_inside_target(obj_pos, obs.maniskill_obs['extra']["right_zone"][env_id]):
            if robot_name == "arm1": return BaseToolsAPI._return_failed_tool(f"Object {object_name} is out of range for robot {robot_name}")
        elif is_object_inside_target(obj_pos, obs.maniskill_obs['extra']["left_zone"][env_id]):
            if robot_name == "arm2": return BaseToolsAPI._return_failed_tool(f"Object {object_name} is out of range for robot {robot_name}")

        
        target_area += "_zone"
        target_pose = obs.maniskill_obs['extra'][target_area][env_id]

        # We must have a final pose to avoid collision between two robots.
        final_pose = None
        if target_area == "mutual_zone":
            if robot_name == "arm1":
                x = -0.2
            else:
                x = 0.2
            
            final_pose = sapien.Pose(
                p=[0, x, 0.3],
                q=[0,1,0,0]
            )

        poses = compute_grasp_drop_trajectory(
            self.get_agent(robot_name), obj_pos, target_pose, 
            drop_seuil=0.05, approach_seuil=0.1, final_pose=final_pose)
        

        def verifier(new_obs: Dict) -> ToolResult:
            # Check if the object is no longer in the gripper and is now in the box
            obj_pose = new_obs["extra"][object_name][env_id]
            if not is_object_inside_target(obj_pose, new_obs["extra"][target_area][env_id]):
                return ToolResult(False, reason=f"{object_name} is not in {target_area}, you can retry")
            return ToolResult(True, reason=f"Successfully depose {object_name} to {target_area}")

        return ToolExecution(poses=poses,verifier=verifier,reason="")
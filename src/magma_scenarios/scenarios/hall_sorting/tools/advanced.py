# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.base.data_structures import ToolExecution, ToolResult
from magma_core.utils.env_utils import is_object_inside_target

from magma_scenarios.utils import compute_grasp_drop_trajectory

from typing import Dict, List
import sapien

from .base import HallTool

class AdvancedHallTool(HallTool):
    @register_tool(
        description="Make the robot move to a specific hall",
        params_spec={
                "target_hall": {"description": "The id of the destination hall", "type":str},
            }
    )
    def move_to(self, obs : Dict, env_id : int, params: Dict) -> ToolExecution:        
        robot_name = obs['selected_robot_name']
        object_name = params['object']
        target_area = params['target_area']

        if not target_area in obs['task_attributes']["known_area"]:
            return BaseToolsAPI._return_failed_tool(f"You used an unknow area : {target_area}.")
        elif not object_name in obs['extra']:
            return BaseToolsAPI._return_failed_tool(f"You used an unknow object name : {object_name}.")

        if robot_name == "arm1" and target_area == "right":
            return BaseToolsAPI._return_failed_tool(f"The area {target_area} is out of range for robot {robot_name}.")
        elif robot_name == "arm2" and target_area == "left":
            return BaseToolsAPI._return_failed_tool(f"The area {target_area} is out of range for robot {robot_name}.")


        obj_pos = obs['extra'][object_name][env_id]
        if is_object_inside_target(obj_pos, obs['extra']["right_zone"][env_id]):
            if robot_name == "arm1": return BaseToolsAPI._return_failed_tool(f"Object {object_name} is out of range for robot {robot_name}")
        elif is_object_inside_target(obj_pos, obs['extra']["left_zone"][env_id]):
            if robot_name == "arm2": return BaseToolsAPI._return_failed_tool(f"Object {object_name} is out of range for robot {robot_name}")

        
        target_area += "_zone"
        target_pose = obs['extra'][target_area][env_id]

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


    @register_tool(
        description="Take all objects specified by ID.",
        params_spec={
                "list_of_objects": {"description": "The id of the destination hall", "type":str},
            }
    )
    def take_objects(self, obs : Dict, env_id : int, params: Dict) -> ToolExecution:        
        robot_name = obs['selected_robot_name']
        object_name = params['object']
        target_area = params['target_area']

        if not target_area in obs['task_attributes']["known_area"]:
            return BaseToolsAPI._return_failed_tool(f"You used an unknow area : {target_area}.")
        elif not object_name in obs['extra']:
            return BaseToolsAPI._return_failed_tool(f"You used an unknow object name : {object_name}.")

        if robot_name == "arm1" and target_area == "right":
            return BaseToolsAPI._return_failed_tool(f"The area {target_area} is out of range for robot {robot_name}.")
        elif robot_name == "arm2" and target_area == "left":
            return BaseToolsAPI._return_failed_tool(f"The area {target_area} is out of range for robot {robot_name}.")


        obj_pos = obs['extra'][object_name][env_id]
        if is_object_inside_target(obj_pos, obs['extra']["right_zone"][env_id]):
            if robot_name == "arm1": return BaseToolsAPI._return_failed_tool(f"Object {object_name} is out of range for robot {robot_name}")
        elif is_object_inside_target(obj_pos, obs['extra']["left_zone"][env_id]):
            if robot_name == "arm2": return BaseToolsAPI._return_failed_tool(f"Object {object_name} is out of range for robot {robot_name}")

        
        target_area += "_zone"
        target_pose = obs['extra'][target_area][env_id]

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
# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.base.data_structures import ToolExecution, ToolResult, Observation
from magma_core.utils.gripper_utils import is_object_in_gripper, find_object_in_gripper

from magma_scenarios.utils import compute_grasp_trajectory

from typing import Dict, List
import sapien, torch


class Tool(BaseToolsAPI):

    @register_tool(
            description="Take a cube from the environment.",
            params_spec={
                "color": {"description": "The color of the cubes to take", "type": str}
            }
    )
    def take_cube(self, obs: Observation, env_id, params : Dict) -> ToolExecution:
        poses = []
        r = ""
        color = params.get("color", None)

        for obj_name, obj_pos in obs.maniskill_obs["extra"].items():
            if color in obj_name:
                poses = compute_grasp_trajectory(self.get_agent(),obj_pos[env_id].cpu().numpy())
                break
        if not poses:
            r=f"No objects corresponding to color = {color}. You must pass the color of the object to take."
        # define verifier inline
        def verifier(new_obs: Dict) -> ToolResult:
            # e.g. check if gripper is holding the right object
            reason=f"No object with {color} color was found. You must pass the color of the object to take."
            ok = False
            for obj_name, obj_pos in new_obs["extra"].items():
                if color in obj_name:
                    ok = is_object_in_gripper(new_obs["extra"]["agent_tcp"][env_id], obj_pos[env_id])   
                    if ok:
                        reason = f"You have a {color} cube in your gripper"
                        break
                    else:
                        reason = f"Failed to grasp the {color} cube. You can retry."
            return ToolResult(ok,reason)
        return ToolExecution(poses=poses, verifier=verifier, reason=r)
    
    @register_tool(
            description="Put the held object in a box.",
            params_spec={}
    )
    def put_to_box(self, obs : Observation, env_id : int, params: Dict) -> ToolExecution:
        r = ""
        obj_in_gripper = None
        poses = []

        def verifier(new_obs: Dict) -> ToolResult:
            # Check if the object is no longer in the gripper and is now in the box
            obj_pose = new_obs["extra"][obj_in_gripper][env_id]
            if is_object_in_gripper(new_obs["extra"]["agent_tcp"][env_id], obj_pose):
                return ToolResult(False, reason=f"The object is still in the gripper")
            dist = torch.norm(obj_pose[:2] - new_obs["extra"]["white_box"][env_id][:2])
            if dist > 0.1:
                return ToolResult(False, reason=f"The object is not in the box and not in the gripper")
            return ToolResult(True)

        if params:
            r = f"Unknow arguments : {', '.join(params.keys())}"
        else:
            reduced_obs = {k: v[env_id][:3] for k, v in obs.maniskill_obs["extra"].items()}
            agent_tcp_pos = reduced_obs.pop("agent_tcp", None)[:3]
            obj_in_gripper = find_object_in_gripper(
                agent_tcp_pos,
                reduced_obs
            )

            if obj_in_gripper is None:
                r = f"There is no object currently in the gripper. You must pick one first."
            else:
                box_pose = obs.maniskill_obs["extra"]["white_box"][env_id].cpu().numpy()
                box_pose[2] += 0.2
                poses = [sapien.Pose(p=box_pose[:3],q=[0,1,0,0]), "OPEN"]

        return ToolExecution(poses=poses, verifier=verifier, reason=r)
    
    @register_tool(
        description="Stack the cube on top of another",
        params_spec={
            "cube_name": {"description": "The name the cube to stack on", "type": str}
        }
    )
    def stack_on(self, obs : Observation, env_id : int, params: Dict) -> ToolExecution:
        r = ""
        obj_in_gripper = None
        location = params.get('cube_name')
        poses = []

        def verifier(new_obs: Dict) -> ToolResult:
            obj_pose = new_obs["extra"][obj_in_gripper][env_id]
            if is_object_in_gripper(new_obs["extra"]["agent_tcp"][env_id], obj_pose):
                return ToolResult(False, reason=f"The object is still in the gripper")
            dist = torch.norm(obj_pose[:2] - new_obs["extra"][location][env_id][:2])
            if dist > 0.1:
                return ToolResult(False, reason=f"The object is not in the box and not in the gripper")
            return ToolResult(True, reason=f"You have successfully placed {obj_in_gripper} to {location}")        

        reduced_obs = {k: v[env_id][:3] for k, v in obs.maniskill_obs["extra"].items()}
        agent_tcp_pos = reduced_obs.pop("agent_tcp", None)[:3]
        obj_in_gripper = find_object_in_gripper(
            agent_tcp_pos,
            reduced_obs
        )

        if obj_in_gripper is None:
            r = f"There is no object currently in the gripper. You must pick one first."
        else:
            obj_pos = obs.maniskill_obs["extra"].get(location,None)
            if obj_pos != None:
                location_pose = obj_pos[env_id].cpu().numpy()
                location_pose[2] += 0.2
                poses = [sapien.Pose(p=location_pose[:3],q=[0,1,0,0]), "OPEN"]
            else:
                r = f"Unknown cube color {location} for location parameter. Please use only known objects."

        return ToolExecution(poses=poses, verifier=verifier, reason=r)
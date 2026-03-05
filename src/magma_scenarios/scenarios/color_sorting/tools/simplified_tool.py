# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.base.data_structures import ToolExecution, ToolResult, Observation
from magma_core.utils.env_utils import is_object_inside_target
from magma_core.utils.gripper_utils import find_object_in_gripper, is_object_in_gripper

from magma_scenarios.utils import compute_grasp_trajectory

from typing import Dict, List
import sapien

class ColorSimplifiedTools(BaseToolsAPI):

    @register_tool(
            description="Take an object corresponding to the color. Automatically take only non-sorted object.",
            params_spec={"color":{"description": "The color of the object to take", "type": str}}
    )
    def take_object_per_color(self, obs: Observation, env_id, params : Dict) -> ToolExecution:
        poses = []
        color = None
        obj_name = None
        r = ""
        box_poses = [obs.maniskill_obs["extra"]["green_box_pose"][env_id],
                     obs.maniskill_obs["extra"]["yellow_box_pose"][env_id]
                     ]

        color = params.get("color", None)

        for obj_name, obj_pos in obs.maniskill_obs["extra"].items():
            if f"{color}_cube" in obj_name:
                if not is_object_inside_target(obj_pos[env_id], box_poses[0]) and \
                    not is_object_inside_target(obj_pos[env_id], box_poses[1]):
                    poses = compute_grasp_trajectory(self.get_agent(), obj_pos[env_id].cpu().numpy())
                    break
        if not poses:
            r=f"No cube corresponding to color = {color} outside the boxes. You must pass the color of an object to take outside the boxes."
        # define verifier inline
        def verifier(new_obs: Dict) -> ToolResult:
            ok = False
            obj_pos = new_obs["extra"][obj_name]
            ok = is_object_in_gripper(new_obs["extra"]["agent_tcp"][env_id], obj_pos[env_id])   
            if ok:
                reason = f"You have a {color} cube in your gripper"
            else:
                reason = f"Failed to grasp the {color} cube. You can retry."
            return ToolResult(ok,reason)
  
        return ToolExecution(poses=poses, verifier=verifier, reason=r)
    
    @register_tool(
            description="Put the held object in a box corresponding to the given color",
            params_spec={"color":{"description": "The color of the target box", "type": str}}
    )
    def put_to_box(self, obs : Observation, env_id : int, params: Dict) -> ToolExecution:
        r = ""
        obj_in_gripper = None
        poses = []

        # Recover box of the requested color
        color = params["color"]
        box_name = f"{color}_box_pose"
        # Check that box of this color exists
        if not box_name in obs.maniskill_obs["extra"]:
            r = f"No {color} box in the scene."
            return ToolExecution(poses=poses, verifier=None, reason=r)
        
        def verifier(new_obs: Dict) -> ToolResult:
            # Check if the object is no longer in the gripper and is now in the box
            obj_pose = new_obs["extra"][obj_in_gripper][env_id]
            if is_object_in_gripper(new_obs["extra"]["agent_tcp"][env_id], obj_pose):
                return ToolResult(False, reason=f"The object is still in the gripper")
            if not is_object_inside_target(obj_pose, new_obs["extra"][box_name][env_id]):
                return ToolResult(False,
                    reason=f"The object is not in the box and not in the gripper")
            return ToolResult(True)

        reduced_obs = {k: v[env_id][:3] for k, v in obs.maniskill_obs["extra"].items()}
        agent_tcp_pos = reduced_obs.pop("agent_tcp", None)[:3]
        obj_in_gripper = find_object_in_gripper(
            agent_tcp_pos,
            reduced_obs
        )

        if obj_in_gripper is None:
            r = f"There is no object currently in the gripper. You must pick one first."
        else:
            box_pose = obs.maniskill_obs["extra"][box_name][env_id].cpu().numpy()
            box_pose[2] += 0.2
            poses = [sapien.Pose(p=box_pose[:3],q=[0,1,0,0]), "OPEN"]

        return ToolExecution(poses=poses, verifier=verifier, reason=r)
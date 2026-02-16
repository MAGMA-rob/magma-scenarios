# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.base.data_structures import ToolExecution, ToolResult
from magma_core.utils.env_utils import is_object_inside_target
from magma_core.utils.gripper_utils import find_object_in_gripper, is_object_in_gripper

from magma_scenarios.utils import compute_grasp_trajectory

from typing import Dict, List
import sapien

class ColorDetectionTools(BaseToolsAPI):

    @register_tool(
            description="Return the description and position of all detected object from the table.",
            params_spec={}
    )
    def get_object_state(self, obs, env_id, params : Dict) -> ToolExecution:

        detected_obj = {"green_box":[], "yellow_box":[], "table":[]}

        for obj_name, obj_pose in obs['extra'].items():
            if "cube" in obj_name:
                if is_object_inside_target(obj_pose[env_id],obs["extra"]["green_box_pose"][env_id]):
                    detected_obj["green_box"].append(obj_name)
                elif is_object_inside_target(obj_pose[env_id],obs["extra"]["yellow_box_pose"][env_id]):
                    detected_obj["yellow_box"].append(obj_name)
                else:
                    detected_obj["table"].append(obj_name)

        def verifier(new_obs: Dict) -> ToolResult:
            s = "This is the position of existing objects: "

            if len(detected_obj['green_box']) == 0:
                s += "green_box is empty. "
            else:
                s += ",".join(detected_obj["green_box"]) + " are in the green_box. "

            if len(detected_obj['yellow_box']) == 0:
                s += "green_box is empty. "
            else:
                s += ",".join(detected_obj['yellow_box']) + " are in the yellow_box. "

            if len(detected_obj['table']) > 0:
                s += ",".join(detected_obj['table']) + " are not sorted."

            return ToolResult(True,s)

        return ToolExecution(poses=["OK"], verifier=verifier)
    
    @register_tool(
            description="Take an object by its name.",
            params_spec={"name": {"description": "The name of the object to take", "type": str}}
    )
    def take_object_per_id(self, obs, env_id, params : Dict) -> ToolExecution:
        poses = []
        obj_name = None
        r = ""

        box_poses = [obs["extra"]["green_box_pose"][env_id],
                     obs["extra"]["yellow_box_pose"][env_id]
                     ]

        obj_name = params.get("name", None)
        obj_pose = obs["extra"].get(obj_name,None)
        if obj_pose != None:
            if not is_object_inside_target(obj_pose[env_id], box_poses[0]) and \
                not is_object_inside_target(obj_pose[env_id], box_poses[1]):
                    poses = compute_grasp_trajectory(self.get_agent(),obj_pose[env_id].cpu().numpy())
            else:
                r=f"{obj_name} is already sorted inside a container. You can not take it."
        else:
            r=f"{obj_name} is not a known objects. You can use get_objects_state to see all detected objects."

        # define verifier inline
        def verifier(new_obs: Dict) -> ToolResult:
            ok = False
            obj_pos = new_obs["extra"][obj_name]
            ok = is_object_in_gripper(new_obs["extra"]["agent_tcp"][env_id], obj_pos[env_id])   
            if ok:
                reason = f"You have {obj_name} in your gripper"
            else:
                reason = f"Failed to grasp {obj_name} due to planning error."
            return ToolResult(ok,reason)

        return ToolExecution(poses=poses, verifier=verifier, reason=r)
    
    @register_tool(
            description="Put the held object in a box corresponding to the given color.",
            params_spec={
                "color": {"description": "The color of the target box", "type": str}
            }
    )
    def put_to_box(self, obs : Dict, env_id : int, params: Dict) -> ToolExecution:
        r = ""

        obj_in_gripper = None
        poses = []

        # Recover box of the requested color
        color = params["color"]
        box_name = f"{color}_box_pose"
        # Check that box of this color exists
        if not box_name in obs["extra"]:
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
            return ToolResult(True, reason=f"Successfuly sent {obj_in_gripper} to {color} box")

        reduced_obs = {k: v[env_id][:3] for k, v in obs["extra"].items()}
        agent_tcp_pos = reduced_obs.pop("agent_tcp", None)[:3]
        obj_in_gripper = find_object_in_gripper(
            agent_tcp_pos,
            reduced_obs
        )

        if obj_in_gripper is None:
            r = f"There is no object currently in the gripper. You must pick one first."
        else:
            box_pose = obs["extra"][box_name][env_id].cpu().numpy()
            box_pose[2] += 0.2
            poses = [sapien.Pose(p=box_pose[:3],q=[0,1,0,0]), "OPEN"]

        return ToolExecution(poses=poses, verifier=verifier, reason=r)
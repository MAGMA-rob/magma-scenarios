# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.utils.env_utils import is_object_inside_target
from magma_core.base.data_structures import Log, ToolExecution, ToolResult

from magma_scenarios.utils import compute_press_trajectory, compute_grasp_drop_trajectory, sapien_to_tensor

from typing import Dict, List
import sapien, torch

class MakingCoffeeTool(BaseToolsAPI):
    """
    Tools to make a coffee.
    """

    button_STROKE = 0.0018

    def is_button_pressed(self, btn_translation) -> bool:
            button_STROKE_LIMIT = self.button_STROKE/2
            return (btn_translation > button_STROKE_LIMIT)

    @register_tool(
            description="Press the start button of the coffee maker.",
            params_spec={}
    )
    def press_button(self, obs, env_id, params : Dict) -> ToolExecution:
        """tool to press a button"""
        poses = []

        poses = compute_press_trajectory(obj_pos=obs["extra"]["coffee_maker"][env_id][:3].cpu().numpy())

        # define verifier inline
        def verifier(new_obs: Dict) -> ToolResult:
            ok = self.is_button_pressed(new_obs["extra"]["coffee_maker"][env_id][-2])   
            if ok:
                reason = f"Coffee launched!"
            else:
                reason = f"Failed to press the coffee maker button."
            return ToolResult(ok,reason,logs=Log(""))
        
        return ToolExecution(poses=poses, verifier=verifier, reason="")
    
    @register_tool(
            description="Load a coffee capsule inside the coffee maker.",
            params_spec={
                "name": {"description": "The name of the capsule to take.", "type": str}
            }
    )
    def load_capsule(self, obs, env_id, params : Dict) -> ToolExecution:
        """load a capsule in the coffee maker"""
        poses = []
        coffee_name = None
        pods_name = None
        target_capsule_pose = obs["add_constants"]["loaded_capsule_pose"]
        base_pose = obs["add_constants"]["base_pose"]

        coffee_name = params.get("name", None)

        target_obj = obs["extra"]["coffee_maker"][env_id].cpu().numpy()
        drop_pose = sapien.Pose(
            p=target_obj[:3] + target_capsule_pose.get_p(),
            q = [0,1,0,0]
            )
        for obj_name, obj_pos in obs["extra"].items():
            if coffee_name in obj_name:
                pods_name = obj_name
                poses = compute_grasp_drop_trajectory(
                    self.get_agent(), obj_pose=obj_pos[env_id].cpu().numpy(), drop_pose=drop_pose,
                    approach_seuil=0.08, drop_approach_pose=base_pose)
                break
        r = ""      
        if not poses:
            r=f"No coffee_pod named {coffee_name}. Use only known pods."
        
        # define verifier inline
        def verifier(new_obs: Dict) -> ToolResult:
            # e.g. check if gripper is holding the right object
            if not pods_name:
                reason = f"No coffee_pod named {coffee_name}. Use only known pods."
                ok = False
            else:
                capsule_target = sapien_to_tensor(target_capsule_pose, new_obs["extra"]["coffee_maker"].device)
                ok = is_object_inside_target(new_obs["extra"][pods_name][env_id],
                                             torch.add(new_obs["extra"]["coffee_maker"][env_id][:7], capsule_target),
                                            0.06)
                if ok:
                    reason = f"You have loaded a {coffee_name} capsule."
                else:
                    reason = f"Failed to load the {coffee_name} capsule. You can retry."

            return ToolResult(ok, reason, logs=Log(""))

        return ToolExecution(poses=poses, verifier=verifier, reason=r)
    
    @register_tool(
            description="Place a mug on the coffee maker.",
            params_spec={}
    )
    def place_mug(self, obs, env_id, params : Dict) -> ToolExecution:
        """drop a mug on the coffee maker"""
        poses = []
        target_obj = []
        mug_pose = []
        target_mug_pose = obs["add_constants"]["dropped_mug_pose"]
        base_pose = obs["add_constants"]["base_pose"]

        target_obj = obs["extra"]["coffee_maker"][env_id].cpu().numpy()

        drop_pose = sapien.Pose(
            p=target_obj[:3] + target_mug_pose.get_p(),
            q = [0,1,0,0]
            )
        
        mug_pose = obs["extra"]["mug"][env_id].cpu().numpy()
        
        poses = compute_grasp_drop_trajectory(
            self.get_agent(), obj_pose=mug_pose[:3], drop_pose=drop_pose,
            final_pose=base_pose, drop_approach_pose=base_pose)
        if not poses:
            r=f"No mug found. Alert user"
                
        # define verifier inline
        def verifier(new_obs: Dict) -> ToolResult:
            # e.g. check if gripper is holding the right object
            mug_target = sapien_to_tensor(target_mug_pose, new_obs["extra"]["coffee_maker"].device)
            ok = is_object_inside_target(
                torch.add(new_obs["extra"]["coffee_maker"][env_id][:7], mug_target),
                new_obs["extra"]["mug"][env_id],
                0.06)
            if ok:
                reason = f"You have placed the mug."
            else:
                reason = f"Failed to place the mug. You can retry."
               
            return ToolResult(ok,reason,logs=Log(""))
  
        return ToolExecution(poses=poses, verifier=verifier, reason="")
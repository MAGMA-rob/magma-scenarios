# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.utils.env_utils import is_object_inside_target
from magma_core.base.data_structures import Log, ToolExecution, ToolResult, Observation

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
    def press_button(self, obs: Observation, env_id, params : Dict) -> ToolExecution:
        """tool to press a button"""
        poses = []

        poses = compute_press_trajectory(obj_pos=obs.maniskill_obs["extra"]["coffee_maker"][env_id][:3].cpu().numpy())

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
    def load_capsule(self, obs: Observation, env_id, params : Dict) -> ToolExecution:
        """load a capsule in the coffee maker"""
        poses = []
        coffee_name = None
        pods_name = None
        target_capsule_pose = obs.add_constants["loaded_capsule_pose"]
        base_pose = obs.add_constants["base_pose"]

        coffee_name = params.get("name", None)

        target_obj = obs.maniskill_obs["extra"]["coffee_maker"][env_id].cpu().numpy()
        drop_pose = sapien.Pose(
            p=target_obj[:3] + target_capsule_pose.get_p(),
            q = [0,1,0,0]
            )
        for obj_name, obj_pos in obs.maniskill_obs["extra"].items():
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

            return ToolResult(ok, reason, logs=Log(coffee_name))

        return ToolExecution(poses=poses, verifier=verifier, reason=r)
    
    @register_tool(
            description="Place a mug on the coffee maker.",
            params_spec={}
    )
    def place_mug(self, obs: Observation, env_id, params : Dict) -> ToolExecution:
        """drop a mug on the coffee maker"""
        poses = []
        target_obj = []
        mug_pose = []
        target_mug_pose = obs.add_constants["dropped_mug_pose"]
        base_pose = obs.add_constants["base_pose"]

        target_obj = obs.maniskill_obs["extra"]["coffee_maker"][env_id].cpu().numpy()

        drop_pose = sapien.Pose(
            p=target_obj[:3] + target_mug_pose.get_p(),
            q = [0,1,0,0]
            )
        
        mug_pose = obs.maniskill_obs["extra"]["mug"][env_id].cpu().numpy()
        
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


    @register_tool(
        description = "get people from a known team",
        params_spec = {"team" : {
                "description" : "name of the team",
                "type" : str 
            }
            }
        )
    def people_from_team(self, obs: Observation, env_id, params : Dict) -> ToolExecution:
        team_name = params.get("team",None)
        teams = obs.add_constants.get("teams",{})
        result = teams.get(team_name,None)
        def verifier(new_obs: Dict)-> ToolResult:
            #check if the people belongs to the team
            if result is None :
                return ToolResult(False,f"team {team_name} doesn't exist",logs=Log(""))
            return ToolResult(True,f"people in {team_name} : {result}",logs=Log(""))

        return ToolExecution(poses=["OK"], verifier=verifier, reason="")
                    

    @register_tool(
        description = "get the peopl team's name",
        params_spec= {"people" : {
                "description" : "name of the person",
                "type" : str
            }}
        )
    def team_from_people(self, obs: Observation, env_id, params : Dict) -> ToolExecution:
        people = params.get("people",None)
        teams = obs.add_constants.get("teams",{})
        result = None
        for teams, team_members in teams.items() :
            if people in team_members :
                result = teams
                break
        def verifier(new_obs:Dict)-> ToolResult:
            if result is None :
                return ToolResult(False,f"person doesn't exist")
            return ToolResult(True,f"{people} in team {result}")

        return ToolExecution(poses = ["OK"], verifier = verifier, reason="") 
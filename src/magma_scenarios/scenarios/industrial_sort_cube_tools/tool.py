# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.base.data_structures import ToolExecution, ToolResult
from magma_core.utils.env_utils import is_object_inside_target

from magma_scenarios.utils import compute_grasp_trajectory

from typing import Dict, List
import sapien, torch

class CycleTool(BaseToolsAPI):

    @register_tool(
            description="Launch a cycle for one reference only to extract them from the bin and send them to a target container.",
            params_spec={
                    "reference": {"description": "The reference to sort.", "type": str},
                    "manufacturing_order" : {"description": "The Manufacturing Order associated with this cycle", "type": str},
                    "target_container" : {"description": "The target container to send objects inside.", "type": str}
                }
    )
    def launch_cycle(self, obs, env_id, params : Dict) -> ToolExecution:
        obj_to_sort = []
        manu_order = ""
        task_attributes = obs['task_attributes']

        def verifier(new_obs: Dict) -> ToolResult:  
            for obj_name in obj_to_sort:
                if not is_object_inside_target(new_obs["extra"][obj_name][env_id], new_obs["extra"][params["target_container"]][env_id]):
                    return ToolResult(False, f"Launch cycle ended but an object of {params['reference']} is still unsorted.")
            return ToolResult(True, f"All {params['reference']} has been sorted to {params['target_container']}", logs=Log(content=manu_order))

        if not params["reference"] in task_attributes["known_reference"]:
            return ToolExecution([], verifier=verifier, reason=f"You used an unknown reference : {params['reference']}. Please use only known reference.")

        if not params["target_container"] in task_attributes["known_target"]:
            return ToolExecution([], verifier=verifier, reason=f"You used an unknown target : {params['target_container']}. Please use only known target.")

        manu_order = params['manufacturing_order']

        for obj_name, obj_pos in obs["extra"].items():
            if (params["reference"] in obj_name 
                and 
                (torch.norm(obj_pos[env_id][:2] - obs["extra"]["containerA"][env_id][:2]) > 0.1
                 or torch.norm(obj_pos[env_id][:2] - obs["extra"]["containerB"][env_id][:2]) > 0.1
                 )):
                obj_to_sort.append(obj_name)

        if not obj_to_sort:
            return ToolExecution([], verifier=verifier, reason=f"All objects with reference {params['reference']} has been sorted.")
        
        
        cpt = 0
        cpt_max = len(obj_to_sort) + 2

        def redo(new_obs: Dict) -> List[sapien.Pose]:
            nonlocal cpt
            cpt +=1
            if cpt > cpt_max:
                print("EXCEEDDING MAX NUMBER OF RETRY")
                return []

            for obj_name in obj_to_sort:
                
                obj_pose = new_obs["extra"][obj_name][env_id]
                if torch.norm(obj_pose[:2] - new_obs["extra"][params["target_container"]][env_id][:2]) < 0.1: # juste sur x , y prcq sinon lobjet a pas le temps de tomber
                    continue


                poses = compute_grasp_trajectory(self.get_agent(), obj_pose.cpu().numpy())
                box_pose = obs["extra"][params["target_container"]][env_id].cpu().numpy()
                box_pose[2] += 0.2
                poses += [sapien.Pose(p=box_pose[:3],q=[0,1,0,0]), "OPEN"]

                return poses

            return []

        p = redo(obs)

        return ToolExecution(poses=p,verifier=verifier,redo=redo)
    
    @register_tool(
            description="Add a new ref to your attributes. The ref must be of exactly four character.",
            params_spec={
                "ref_name" : {"description" : "The name of the ref to add. It must be composed of 4 character only.", "type":str}
            }
    )
    def add_ref(self, obs, env_id, params : Dict) -> ToolExecution:

        task_attributes = obs['task_attributes']
        ref_name = params['ref_name']

        def verifier(new_env_state : Dict) -> ToolResult:
            return ToolResult(True, f"Reference {ref_name} successfully added", logs=Log(content=("known_reference",ref_name), action="ADD"))

        if len(ref_name) != 4:
            return ToolExecution(poses=[], verifier=verifier, reason="The reference must be 4 character long.")
        
        if ref_name in task_attributes["known_reference"]:
            return ToolExecution(poses=[], verifier=verifier, reason=f"The ref {ref_name} already exists in memory.")
        
        task_attributes["known_reference"].append(ref_name)

        return ToolExecution(poses=["OK"], verifier=verifier)
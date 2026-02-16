# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.base.data_structures import ToolExecution, ToolResult
from magma_core.base.data_structures import Log
from magma_core.utils.env_utils import is_object_inside_target

from magma_scenarios.utils import compute_grasp_trajectory

from typing import Dict, List
import sapien

class CycleTool(BaseToolsAPI):

    @register_tool(
        description="Launch a delivery cycle under a specific manufacturing order. The cycle can produce a fixed number of deliveries or run indefinitely if delivery_number is not set. Each delivery follows the given recipe.",
        params_spec={
            "manufacturing_order": {
                "type": str,
                "description": "The manufacturing order associated with the cycle."
            },
            "delivery_number": {
                "type": int,
                "description": "The number of deliveries to make."
            },
            "recipe": {
                "type": list,
                "description": "A list of product names to include in each delivery box. Repeat items in the list to include multiple instances of the same product."
            }
        },
        optional = ["delivery_number"]
    )
    def launch_cycle(self, obs : Dict, env_id : int, params: Dict) -> ToolExecution:
        obj_to_sort = []
        n = 0
        manu_order = ""
        
        if len(params['recipe']) == 0:
            return ToolExecution(poses=[], verifier=None, reason="The recipe is empty.")
        
        n = params.get('delivery_number',-1)
        if n == 0:
            return ToolExecution(
                [],verifier=None,
                reason="delivery_number can not be equal to 0. You can set it to -1 for infinite cycle.")
        manu_order = params['manufacturing_order']
        
        def verifier(new_obs: Dict) -> ToolResult:
            for obj in obj_to_sort:
                if not is_object_inside_target(new_obs['extra'][obj][env_id], new_obs["extra"]["container"][env_id], keep_tensor=False):
                    return ToolResult(False, reason=f"The cycle fail. At least one object is not in the delivery.")
            return ToolResult(
                True, 
                reason=f"You have successfully made {n} delivery with recipe : {params['recipe']}", 
                logs=Log(content=(n,manu_order)))
                
            
        for k in params['recipe']:
            if not k in obs['task_attributes']['product_type']:
                return ToolExecution(poses=[], verifier=None, reason=f"Object {k} in recipe is unknow. Please use only object from attributes.")
            select_obj = None
            for obj in obs["extra"]:
                if obj in obj_to_sort: continue #already select    
                if k in obj:
                    select_obj = obj
                    break
            if not select_obj:
                return ToolExecution(poses=[], verifier=None, reason=f"It can only be 2 instance of {k} in each delivery.")
            
            obj_to_sort.append(select_obj)            
            
        unsort_obj = obj_to_sort.copy()

        def redo(new_obs : Dict) -> List[sapien.Pose]:
            poses = []
            if len(unsort_obj) == 0:
                return []
            
            obj = unsort_obj.pop(0)
            obj_pose = new_obs['extra'][obj][env_id]
            poses = compute_grasp_trajectory(self.get_agent(),obj_pose.cpu().numpy())
            box_pose = obs["extra"]["container"][env_id].cpu().numpy()
            box_pose[2] += 0.2
            poses += [sapien.Pose(p=box_pose[:3],q=[0,1,0,0]), "OPEN"]
            return poses
            
        p = redo(obs)

        return ToolExecution(poses=p, verifier=verifier, redo=redo)
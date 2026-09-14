# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.simulation.tools import BaseToolsAPI, register_tool
from magma_core.simulation.data_structures import (
    Observation,
    ToolErrorSupport,
    ToolExecution,
    ToolResult,
)
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_core.simulation.utils.gripper_utils import is_object_in_gripper, find_object_in_gripper
from magma_core.simulation.data_structures import Log

from magma_scenarios.utils import compute_grasp_trajectory, compute_drop_trajectory
from magma_scenarios.templates.errors import OneShotToolFailureError

from .att import AREAS

from typing import Dict
import torch

class WarehouseSortingTool(BaseToolsAPI):

    def _is_object_sorted(
            self,
            obj_pose : torch.Tensor,
            obs_extra: Dict,
            env_id: int,
        ) -> bool:
        for target in AREAS:
            if is_object_inside_target(
                obj_pose[env_id],
                obs_extra[target][env_id],
                keep_tensor=False
            ):
                return True
        return False

    def take_obj(self, obs : Observation, env_id, params : Dict) -> ToolExecution:
        poses = []
        name_obj = None
        selected_object = None
        r = ""

        name_obj = params.get("obj", None)

        for obj_name, obj_pos in obs.maniskill_obs["extra"].items():
            if name_obj in obj_name:
                if is_object_in_gripper(
                    obs.maniskill_obs["extra"]["agent_tcp"][env_id],
                    obj_pos[env_id],
                    threshold=0.05,
                ):
                    return ToolExecution(
                        poses=[],
                        verifier=None,
                        reason=f"The object {name_obj} is already in the gripper.",
                    )

                if self._is_object_sorted(obj_pos, obs.maniskill_obs["extra"], env_id):
                    return ToolExecution(poses=[],
                                         verifier=None,
                                         reason=f"The object {name_obj} is already sorted. It can not be taken again.")

                selected_object = obj_name
                poses = compute_grasp_trajectory(self.get_agent(),obj_pos[env_id].cpu().numpy())
                break
        if not poses:
            r=f"No objects names corresponding to {name_obj}. You must pass the name of the object to take."

        # define verifier inline
        def verifier(new_obs: Dict) -> ToolResult:
            # e.g. check if gripper is holding the right object
            reason=f"No object with {name_obj} name was found. You must pass the name of the object to take."
            ok = False
            if selected_object is not None:
                obj_pos = new_obs["extra"][selected_object]
                ok = is_object_in_gripper(
                    new_obs["extra"]["agent_tcp"][env_id],
                    obj_pos[env_id],
                    threshold=0.05,
                )
                if ok:
                    reason = f"You have a {name_obj} object in your gripper"
                else:
                    reason = f"Failed to grasp the {name_obj} object. You can retry."
            return ToolResult(ok, reason)
        return ToolExecution(
            poses=poses,
            verifier=verifier,
            reason=r,
            allowed_moving_actors=(
                [selected_object] if selected_object is not None else None
            ),
        )
    

    def depose(self, obs : Observation, env_id : int, params: Dict) -> ToolExecution:
        obj_in_gripper = None
        poses = []
        r = ""
        area_name = params.get("target", None)

        def verifier(new_obs: Dict) -> ToolResult:
            # Check if the object is no longer in the gripper and is now in the area
            obj_pose = new_obs["extra"][obj_in_gripper][env_id]
            if is_object_in_gripper(new_obs["extra"]["agent_tcp"][env_id], obj_pose):
                return ToolResult(False, reason=f"The object is still in the gripper")
            if not is_object_inside_target(obj_pose, new_obs["extra"][area_name][env_id], keep_tensor=False):
                return ToolResult(False, reason=f"The object is not in the box and not in the gripper")
            return ToolResult(True, reason=f"Successfully depose {obj_in_gripper} in {area_name}")
        
        reduced_obs = {k: v[env_id][:3] for k, v in obs.maniskill_obs["extra"].items()}
        agent_tcp_pos = reduced_obs.pop("agent_tcp", None)
        obj_in_gripper = find_object_in_gripper(
            agent_tcp_pos,
            reduced_obs
        )

        if obj_in_gripper is None:
            r = f"There is no object currently in the gripper. You must pick one first."
        else:
            if not area_name in obs.maniskill_obs["extra"]:
                r = f"Unknown area {area_name}. Please use only known area."
            else:
                poses = compute_drop_trajectory(self.get_agent(), drop_pose=obs.maniskill_obs["extra"][area_name][env_id].cpu().numpy(),
                                                drop_seuil=0.3, approach_seuil=0.2)

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            reason=r,
            allowed_moving_actors=(
                [obj_in_gripper] if obj_in_gripper is not None else None
            ),
        )
    
    def add_new_location(self, obs : Observation, env_id : int, params: Dict) -> ToolExecution:
        poses = []
        location_name = params["location_name"]
        r = ""

        def verifier(new_obs: Dict) -> ToolResult:
            return ToolResult(True, reason=f"{location_name} was added to known areas" ,logs=Log(content=("target_areas",location_name),action="ADD"))
        
        if location_name in obs.task_attributes['target_areas']:
            r = f"{location_name} already exist! If you want to create a new area, please choose a non-existing name."
        else:
            poses = ["OK"]
            
        return ToolExecution(poses=poses, verifier=verifier, reason=r)
    
    
    def remove_location(self, obs : Observation, env_id : int, params: Dict) -> ToolExecution:
        poses = []
        location_name = params["location_name"]
        r = ""

        def verifier(new_obs: Dict) -> ToolResult:
            return ToolResult(True, reason=f"{location_name} was removed from known areas" ,logs=Log(content=("target_areas",location_name), action="REMOVE"))
        
        if location_name not in obs.task_attributes['target_areas']:
            r = f"{location_name} does not exist! Only existing area can be deleted."
        else:
            poses = ["OK"]
            
        return ToolExecution(poses=poses, verifier=verifier, reason=r)


class WithoutManufacturingOrder(WarehouseSortingTool):

    @register_tool(
            description="Take an unsorted object.",
            params_spec={
                "obj": {"description": "Name of the object to take.", "type": str},
            },
            errors=[
                ToolErrorSupport(
                    OneShotToolFailureError,
                    pre=True,
                    post=False,
                )
            ],
    )
    def take_obj(self, obs, env_id, params: Dict) -> ToolExecution:
        return super().take_obj(obs, env_id, params)
    
    @register_tool(
            description="Put the held object in a storage area.",
            params_spec={
                "target": {"description": "Name of the destination area.", "type": str}
            },
            errors=[
                ToolErrorSupport(
                    OneShotToolFailureError,
                    pre=True,
                    post=False,
                )
            ],
    )
    def depose(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        return super().depose(obs, env_id, params)

    @register_tool(
            description="Add a storage location.",
            params_spec={
                "location_name": {"description": "Name of the new location.", "type": str}
            }
    )
    def add_new_location(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        return super().add_new_location(obs, env_id, params)

    @register_tool(
            description="Remove a storage location.",
            params_spec={
                "location_name": {"description": "Name of the location to remove.", "type": str}
            }
    )
    def remove_location(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        return super().remove_location(obs, env_id, params)

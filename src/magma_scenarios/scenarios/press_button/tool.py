# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.base.data_structures import Log, ToolExecution, ToolResult, Observation

from typing import Dict, List
import sapien, torch
import numpy as np

BTN_STROKE = 0.011

class Tool(BaseToolsAPI):

    def _compute_seq_push_poses(self, obj_pos: np.ndarray) -> List[sapien.Pose]:
        """ Compute action sequence to push an object, we suppose that the wrench is open by default. """
        SEUIL = 0.05
        button_HEIGHT = 0.01

        # pose to move above the object
        above_obj_pos = sapien.Pose(
            p=obj_pos + [0,0,obj_pos[2]+SEUIL],
            q = [0,1,0,0]
            )

        # poses to start and stop pushing
        start_push_pos = sapien.Pose(
            p=obj_pos + [0,0,obj_pos[2]+button_HEIGHT/2],
            q = [0,1,0,0]
            )
        end_push_pos = sapien.Pose(
            p=obj_pos + [0,0,obj_pos[2]+button_HEIGHT/2-BTN_STROKE],
            q = [0,1,0,0]
            )

        # robot pose sequence to perform the task
        return ["CLOSE", above_obj_pos, start_push_pos, end_push_pos, above_obj_pos]

    @register_tool(
            description="Press a button.",
            params_spec={
                "id": {"description": "The name of the button to press", "type": str}
            }
    )
    def press_button(self, obs : Observation, env_id, params : Dict) -> ToolExecution:
        """tool to press a button"""
        poses = []
        id = params["id"]
        btn_pose = obs.maniskill_obs["extra"].get(id, None)
        r = ""
        if btn_pose is None:
            r=f"No objects corresponding to id = {id}. You must pass the id of the object to take."
        else:
            poses = self._compute_seq_push_poses(btn_pose[env_id][:3].cpu().numpy())
        
        # define verifier inline
        def verifier(new_obs: Dict) -> ToolResult:
            reason=f"No object with {id} id was found. You must pass the id of the button to press."
            ok = new_obs["extra"][id][env_id][-1] < -BTN_STROKE/2
            if ok:
                reason = f"You have the {id} button pressed."
                return ToolResult(ok,reason,logs=Log(content=id))
            reason = f"Failed to press the {id} button. You can retry."
            return ToolResult(ok,reason)
        
        return ToolExecution(poses=poses, verifier=verifier, reason=r)
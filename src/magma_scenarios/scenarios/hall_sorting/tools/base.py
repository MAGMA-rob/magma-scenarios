# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.base.data_structures import ToolExecution, ToolResult
from magma_core.utils.env_utils import is_object_inside_target

from typing import Dict, List
import sapien

class HallTool(BaseToolsAPI):

    @register_tool( 
        description="Return the list of detected object in each hall. Can be used with any robots, to get all informations.",
        params_spec={}
    )
    def detect_object(self, obs : Dict, env_id : int, params: Dict) -> ToolExecution:

        def verifier(new_obs: Dict) -> ToolResult:
            # Check if the object is no longer in the gripper and is now in the box
            halls = {'Hall1':[],'Hall2':[],'Hall3':[],'Hall4':[]}
            for obj_name, obj_pose in new_obs['extra'].items():
                if "crate" in obj_name:
                    for hall in halls:
                        if is_object_inside_target(obj_pose[env_id], obs['extra'][hall][env_id], thresh=0.2):
                            halls[hall].append(obj_name)
                            break

            s = "This is the position of existing objects: "

            for hall, objects_list in halls.items():
                if len([hall]) > 0:
                    s += ",".join(objects_list) + f" are in the {hall}. "

            return ToolResult(True,s)

        return ToolExecution(poses=["OK"],verifier=verifier,reason="")

# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.simulation.tools import BaseToolsAPI, register_tool
from magma_core.simulation.data_structures import (
    ToolErrorSupport,
    ToolExecution,
    ToolResult,
    Observation,
)
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_scenarios.utils import compute_grasp_drop_trajectory
from magma_scenarios.templates.errors import (
    OneShotToolFailureError,
    RequestedObjectGraspFailureError,
)
from magma_scenarios.templates.tools import PlacementGrid
from typing import Dict
import sapien
from .brs_attributes import ZONES, sorting_objects
INTACT_STATE = 1
DAMAGED_STATE = 0
STATIC_STATE = -1
SORTING_OBJECT_NAMES = {
    object_name
    for object_names in sorting_objects.values()
    for object_name in object_names
}

class BiRobotTools(BaseToolsAPI):

    placement_grid = PlacementGrid(
        name="bi_robot_sorting_zone",
        rows=3,
        columns=3,
        cell_spacing=0.1,
        # Zone poses use a 180-degree X rotation. This local-cell order keeps
        # the historical world-space preference from (0, -1) through (-1, -1).
        selection_order=[7, 4, 1, 5, 0, 3, 8, 2, 6],
    )

    right_secure_pose = sapien.Pose(
        p=[-0.6, -0.6, 0.16],
        q=[0,0,1,0]
    )

    left_secure_pose = sapien.Pose(
        p=[-0.6, 0.6, 0.16],
        q=[0,0,1,0]
    )

    @register_tool( 
        description="Return all known objects grouped by area and indicate which objects are damaged.",
        params_spec={},
        is_detection=True,
        errors=[
            ToolErrorSupport(
                OneShotToolFailureError,
                pre=True,
                post=False,
            ),
        ],
    )
    def get_objects_state(self, obs : Observation, env_id : int, params: Dict) -> ToolExecution:
        robot_name = obs.selected_robot_name

        if robot_name != "arm1":
            return BaseToolsAPI._return_failed_tool(
                f"{robot_name} cannot inspect objects. Only arm1 has the object inspection skill."
            )

        def verifier(new_obs: Dict) -> ToolResult:
            zones = {'mutual':[],'right':[],'left':[],'damaged':[]}
            for obj_name, obj_data in new_obs['extra'].items():
                obj_pose = obj_data['pose']
                obj_state = obj_data['state']
                if obj_name in ["right_zone","mutual_zone","left_zone","left_arm_tcp","right_arm_tcp"] :
                    continue
                if is_object_inside_target(obj_pose[env_id], new_obs['extra']['right_zone']["pose"][env_id],0.2):
                    zones['right'].append(obj_name)
                elif is_object_inside_target(obj_pose[env_id], new_obs['extra']['left_zone']["pose"][env_id],0.2):
                    zones['left'].append(obj_name)
                elif is_object_inside_target(obj_pose[env_id], new_obs['extra']['mutual_zone']["pose"][env_id],0.2):
                    zones['mutual'].append(obj_name)
                if obj_state[env_id] == DAMAGED_STATE :
                    zones["damaged"].append(obj_name)


            s = "This is the position of existing objects: "

            if len(zones['right']) > 0:
                s += ", ".join(zones["right"]) + " are in the right area. "

            if len(zones['left']) > 0:
                s += ", ".join(zones['left']) + " are in the left area. "

            if len(zones['mutual']) > 0:
                s += ", ".join(zones['mutual']) + " are in the mutual zone. "

            if len(zones['damaged']) > 0:
                s += "Damaged objects: " + ", ".join(zones["damaged"]) + ". "


            return ToolResult(True,s)

        return ToolExecution(poses=["OK"],verifier=verifier,reason="")
    
    @register_tool(
        description="Make the robot take the object and send it to area. The area and the object must be in the range of the robot or it will raise an error.",
        params_spec={
                "object": {"description": "The name of the object to take", "type":str},
                "target_area": {"description": "The target area where to depose the object.", "type":str}
            },
        execution_group="mutual_zone",
        errors=[
            ToolErrorSupport(
                RequestedObjectGraspFailureError,
                pre=True,
                post=False,
            ),
            ToolErrorSupport(
                OneShotToolFailureError,
                pre=True,
                post=False,
            ),
        ],
    )
    def deplace(self, obs : Observation, env_id : int, params: Dict) -> ToolExecution:        
        robot_name = obs.selected_robot_name
        object_name = params['object']
        target_area = params['target_area']

        if not target_area in ZONES:
            return BaseToolsAPI._return_failed_tool(f"You used an unknow area : {target_area}.")
        elif not object_name in obs.maniskill_obs['extra']:
            return BaseToolsAPI._return_failed_tool(f"You used an unknow object name : {object_name}.")

        if robot_name == "arm1" and target_area == "right_zone":
            return BaseToolsAPI._return_failed_tool(f"The area {target_area} is out of range for robot {robot_name}.")
        elif robot_name == "arm2" and target_area == "left_zone":
            return BaseToolsAPI._return_failed_tool(f"The area {target_area} is out of range for robot {robot_name}.")

        target_zone = obs.maniskill_obs['extra'][target_area]["pose"][env_id]

        obj_pos = obs.maniskill_obs['extra'][object_name]["pose"][env_id]
        
        if is_object_inside_target(obj_pos, target_zone, 0.25):
            return BaseToolsAPI._return_failed_tool(f"{object_name} is already in {target_area}.")


        if robot_name == "arm1" and is_object_inside_target(
            obj_pos,
            obs.maniskill_obs['extra']["right_zone"]["pose"][env_id],
            thresh=0.2,
        ):
            return BaseToolsAPI._return_failed_tool(
                f"Object {object_name} is out of range for robot {robot_name}"
            )
        elif robot_name == "arm2" and is_object_inside_target(
            obj_pos,
            obs.maniskill_obs['extra']["left_zone"]["pose"][env_id],
            thresh=0.2,
        ):
            return BaseToolsAPI._return_failed_tool(
                f"Object {object_name} is out of range for robot {robot_name}"
            )

        
        object_poses = {
            name: entry["pose"][env_id]
            for name, entry in obs.maniskill_obs["extra"].items()
            if name in SORTING_OBJECT_NAMES
        }
        target_cell = self.placement_grid.allocate(
            center_pose=target_zone,
            object_poses=object_poses,
            batch_context=obs.tool_batch_context,
            owner=object_name,
            reservation_namespace=f"bi_robot_sorting:{target_area}",
            excluded_objects={object_name},
        )
        if target_cell is None:
            r = f"There is no free space in {target_area}"
            return ToolExecution(poses = [], verifier= None, reason = r)
        target_position = target_cell.world_position.cpu().numpy().tolist()
        
        if robot_name == "arm1":
            security_pose = self.right_secure_pose
        else:
            security_pose = self.left_secure_pose

        transfer_pose = sapien.Pose(
            p=[
                security_pose.p[0],
                security_pose.p[1],
                0.45,
            ],
            q=[0, 1, 0, 0],
        )

        drop_approach_pose = sapien.Pose(
            p=[
                target_position[0],
                target_position[1],
                target_position[2] + 0.25,
            ],
            q=[0, 1, 0, 0],
        )

        target_pose = target_position + [0, 1, 0, 0]
        poses = compute_grasp_drop_trajectory(
            self.get_agent(robot_name), obj_pos, target_pose, 
            drop_seuil=0.05,
            approach_seuil=0.1,
            transfer_pose=transfer_pose,
            drop_approach_pose=drop_approach_pose,
            final_pose=security_pose,
        )
        

        def verifier(new_obs: Dict) -> ToolResult:
            # Check if the object is no longer in the gripper and is now in the box
            obj_pose = new_obs["extra"][object_name]["pose"][env_id]
            if not is_object_inside_target(obj_pose, new_obs["extra"][target_area]["pose"][env_id],thresh=0.2):
                return ToolResult(False, reason=f"{object_name} is not in {target_area}, you can retry")
            return ToolResult(True, reason=f"Successfully depose {object_name} to {target_area}")

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            reason="",
            context={"target_name": object_name},
            allowed_moving_actors=[object_name],
        )


 

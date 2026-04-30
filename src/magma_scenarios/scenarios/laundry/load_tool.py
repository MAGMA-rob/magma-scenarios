# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.base.data_structures import ToolExecution, ToolResult, Observation, ToolErrorSupport
from magma_core.utils.env_utils import is_object_inside_target
from magma_core.utils.gripper_utils import find_object_in_gripper, is_object_in_gripper

from magma_scenarios.utils import compute_grasp_trajectory, compute_drop_trajectory
from magma_core.base.data_structures import Log
from magma_scenarios.envs.laundry.observation import ObjectObservation
from magma_scenarios.scenarios.laundry.attributes import all_detergents, all_clothes
from typing import Dict, List
import sapien, torch

from .laundry_errors import GraspClothesFailureError

LaundryExtraState = dict[str, ObjectObservation]

class LaunchTool(BaseToolsAPI):
    """
    Tools for making a laundry task
    """

    @register_tool(
            description="Pick a product by its name.",
            params_spec={
                "name": {
                    "type": str,
                    "description": "Name of the object to grab.",
                }
            },
            errors=[
                ToolErrorSupport(GraspClothesFailureError, pre=True, post=False)
            ]
    )
    def take(self, obs: Observation, env_id: int, params: dict) -> ToolExecution:
        """Go fetch an object by its name."""

        extra: LaundryExtraState = obs.maniskill_obs["extra"]
        name = params["name"]
        object = extra.get(name, None)
        if object is None:
            return ToolExecution(
                poses=[], verifier=None, reason=f"No object with name {params['name']}"
            )
        poses = compute_grasp_trajectory(self.get_agent(),object["pose"][env_id].cpu().numpy())

        def verifier(new_obs: dict) -> ToolResult:
            """The object must be in the gripper."""
            new_extra: LaundryExtraState = new_obs["extra"]
            object = new_extra.get(name, None)
            if object is None:
                return ToolResult(False, f"The object {name} does not exist anymore.")
            if is_object_in_gripper(
                new_extra["agent_tcp"]["pose"][env_id],
                object["pose"][env_id],
                threshold=0.05 if name == "detergent" else 0.005,
            ):
                return ToolResult(True, f"You have {name} in your gripper.")
            else:
                return ToolResult(
                    False, f"You failed to take the object {name}. You can retry."
                )

        return ToolExecution(poses, verifier=verifier, context={"target_name":name})


    @register_tool(
            description="Put the held clothes into the washing machine.",
            params_spec={}
    )
    def drop(self, obs: Observation, env_id: int, params: dict) -> ToolExecution:
        """Put the object in the gripper into a container."""

        extra = obs.maniskill_obs["extra"]

        target_pos: torch.Tensor = extra["washing_machine"]["pose"][env_id]

        reduced_env = dict((k, v["pose"][env_id][:3]) for k, v in extra.items())
        agent_tcp_position = reduced_env.pop("agent_tcp")
        obj_in_gripper = find_object_in_gripper(
            agent_tcp_position, reduced_env
        )

        if obj_in_gripper is None:
            return ToolExecution(
                poses=[], verifier=None, reason="No clothes in gripper."
            )
        
        poses = [sapien.Pose(agent_tcp_position[:3].cpu().numpy() + (0, 0, 0.1), (0, 1, 0, 0))]
        print("target_pose:", target_pos)
        poses.extend(compute_drop_trajectory(
            self.get_agent(),
            drop_pose=target_pos,
            drop_seuil=0.2,
            approach_seuil=0.3
        ))

        def verifier(new_obs: dict) -> ToolResult:
            """The object must be in the container and not in the gripper."""
            extra = new_obs["extra"]
            obj_pose = extra[obj_in_gripper]["pose"][env_id]
            if is_object_in_gripper(
                extra["agent_tcp"]["pose"][env_id], obj_pose
            ):
                return ToolResult(False, reason="The object is still in the gripper")
            if not is_object_inside_target(obj_pose, extra["washing_machine_basket"]["pose"][env_id]):
                return ToolResult(
                    False, reason="The object is not in the wash machine and not in the gripper"
                )
            return ToolResult(True, reason=f"Successfully put {obj_in_gripper} in the wash machine")

        return ToolExecution(poses, verifier)

    @register_tool(
            description="Wash all clothes in the washing machine, only if a detergent is also in the machine.",
            params_spec={}
    )
    def action_wash(self, obs: Observation, env_id: int, params: dict):
        """Wash the clothes in the machine.

        The machine must contains the detergent.
        This function emits a log with the list of items in the machine."""

        def verifier(new_obs: dict):
            extra = new_obs["extra"]
            used_detergent = []
            for detergent in all_detergents : 
                if is_object_inside_target(extra[detergent]["pose"][env_id], extra["washing_machine_basket"]["pose"][env_id]):
                    used_detergent.append(detergent)
                    
            if len(used_detergent) == 0:
                return ToolResult(False, reason="No detergent in machine.")

            cleaned_items = []
            for obj_name in extra:
                if obj_name in all_detergents or obj_name in ["washing_machine_basket" ,"agent_tcp"]:
                    continue
                if is_object_inside_target(extra[obj_name]["pose"][env_id], extra["washing_machine_basket"]["pose"][env_id]):
                    cleaned_items.append(obj_name)

            if len(cleaned_items) == 0:
                        return ToolResult(False, reason="There is no clothes in the machine!")

            s = ",".join(cleaned_items)
            return ToolResult(True, reason=f"You have washed {s}",logs=Log(content={"clothes" : cleaned_items,"detergent" : used_detergent}))

            

        return ToolExecution(["OK"], verifier)
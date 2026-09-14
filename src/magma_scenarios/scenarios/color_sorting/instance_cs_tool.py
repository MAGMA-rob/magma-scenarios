from typing import Dict

from magma_core.simulation.data_structures import (
    Log,
    Observation,
    ToolErrorSupport,
    ToolExecution,
    ToolResult,
)
from magma_core.simulation.tools import register_tool
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_core.simulation.utils.gripper_utils import is_object_in_gripper
from magma_scenarios.utils import compute_grasp_trajectory
from magma_scenarios.templates.errors import OneShotToolFailureError
from .color_sorting_errors import GraspCubeFailureError, MaskRemainingCubesError
from .cs_tool import ColorDetectionTools

class InstanceColorTools(ColorDetectionTools):
    """
    Color-sorting tools exposing exact cube instance names.
    """

    put = ColorDetectionTools.put

    @register_tool(
        description=("List cube instances on the table and in each tray."),
        params_spec={},
        is_detection=True,
        errors=[ToolErrorSupport( MaskRemainingCubesError, pre=False, post=True)],
    )
    def get_object_state(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]

        detected_objects = {"table": []}

        tray_names = self._get_tray_pose_names(obs)

        for tray_name in tray_names:
            detected_objects[tray_name] = []

        for object_name, object_pose in extra.items():
            if "_cube_" not in object_name:
                continue

            selected_location = "table"

            for tray_name in tray_names:
                if is_object_inside_target(
                    object_pose[env_id],
                    extra[tray_name][env_id],
                    thresh=0.3,
                    keep_tensor=False,
                ):
                    selected_location = tray_name
                    break

            detected_objects[selected_location].append(object_name)

        def verifier(new_obs: Dict) -> ToolResult:
            descriptions = []

            for location, object_names in (detected_objects.items()):
                if not object_names:
                    descriptions.append(f"{location} contains no cube")
                    continue

                descriptions.append(
                    f"{location} contains "
                    + ", ".join(object_names)
                )

            return ToolResult(
                True,
                reason=". ".join(descriptions) + ".",
                context=detected_objects,
                logs=Log(""),
            )

        return ToolExecution(poses=["OK"], verifier=verifier,)

    @register_tool(
        description=("Take a specific cube instance."),
        params_spec={
            "name": {
                "description": ("The exact name of the cube to take."),
                "type": str,
            }
        },
        errors=[
            ToolErrorSupport(OneShotToolFailureError, pre=True, post=False),
            ToolErrorSupport(MaskRemainingCubesError,pre=True, post=False),
            ToolErrorSupport(GraspCubeFailureError, pre=True, post=False),
        ],
    )
    def take(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]
        object_name = params["name"]

        if (object_name not in extra or "_cube_" not in object_name):
            return ToolExecution(
                poses=[],
                verifier=None,
                reason=(f"There is no cube instance named {object_name}."),
            )

        object_pose = extra[object_name]

        if is_object_in_gripper(
            extra["agent_tcp"][env_id],
            object_pose[env_id],
            threshold=0.02,
        ):
            return ToolExecution(
                poses=[],
                verifier=None,
                reason=f"The object {object_name} is already in the gripper.",
            )

        poses = compute_grasp_trajectory(self.get_agent(), object_pose[env_id].cpu().numpy())

        def verifier(new_obs: Dict) -> ToolResult:
            new_object_pose = new_obs["extra"][object_name][env_id]

            object_is_grasped = is_object_in_gripper(
                new_obs["extra"]["agent_tcp"][env_id],
                new_object_pose,
                threshold=0.02,
            )

            if object_is_grasped:
                return ToolResult(True, reason=(f"You have {object_name} in your gripper."))

            return ToolResult(False, reason=(f"Failed to grasp {object_name} due to a planning error."))

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            context={"target_name": object_name},
            allowed_moving_actors=[object_name],
        )

from typing import Dict
import sapien

from magma_core.simulation.data_structures import (
    EnvStateUpdate,
    Log,
    Observation,
    ToolErrorSupport,
    ToolExecution,
    ToolResult,
)
from magma_core.simulation.tools import BaseToolsAPI, register_tool
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_core.simulation.utils.gripper_utils import is_object_in_gripper

from magma_scenarios.utils import (
    compute_drop_trajectory,
    compute_grasp_trajectory,
    compute_swipe_trajectory,
)

from ..common.attributes import ADVANCED_LOCATIONS, CLEAN_STATE, DIRTY_STATE, cleaning_objects, dishware, food
from ..common.errors import GraspItemsFailureError
from ..common.tool_helpers import TableCleaningToolHelpers
from .errors import AdvancedMaskItemsError
from magma_scenarios.templates.errors import OneShotToolFailureError



TABLE_ROBOT = "table_robot"
DISH_ROBOT = "dish_robot"

EXECUTION_GROUP_BY_LOCATION = {
    "table": "shared_workspace",
    "drying_zone": "drying_zone",
}

TCP_KEYS = {
    TABLE_ROBOT: "table_arm_tcp",
    DISH_ROBOT: "dish_arm_tcp",
}

ROBOT_POSE = {
    TABLE_ROBOT: sapien.Pose(
        p=[-7.0000e-01,  6.6501e-01,  1.6978e-01],
        q=[-9.3028e-08,  7.0711e-01,
         -7.0711e-01,  8.6920e-08], 
    ),

    DISH_ROBOT: sapien.Pose(
        p=[-5.0757e-08, -1.3150e+00,  1.6978e-01],
        q=[-9.3028e-08,  7.0711e-01,
         -7.0711e-01,  8.6920e-08], 
    )
}


class AdvancedCleaningTools(BaseToolsAPI, TableCleaningToolHelpers):
    table_center = [-0.1, -0.2, 0]

    def _robot_context(self, obs: Observation):
        robot_name = obs.selected_robot_name
        tcp_key = TCP_KEYS.get(robot_name)

        if tcp_key is None:
            return None, None

        return robot_name, tcp_key

    @register_tool(
        description="Returns visible objects, their locations, and dirty objects.",
        params_spec={},
        is_detection=True,
        errors=[
            ToolErrorSupport(AdvancedMaskItemsError, pre=False, post=True),
            ToolErrorSupport(OneShotToolFailureError, pre=True, post=False),
        ],
    )
    def scan_scene(self, obs: Observation, env_id: int, params: dict) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]

        detected_obj = {
            "table": {},
            "trashcan": {},
            "drying_zone": {},
            "sink": {},
            "food_storage": {},
            "dish_storage": {},
        }

        dirty_objects = []

        ignored_objects = {
            "dish_arm_tcp",
            "table_arm_tcp",
            "drying_zone",
            "food_storage",
            "dish_storage",
            "trashcan",
            "sink",
            "table",
        }

        for name, data in extra.items():
            if name in ignored_objects:
                continue

            if name not in [*food, *dishware, *cleaning_objects]:
                continue

            pos = data["pose"][env_id][:3]
            x, y, _ = pos

            if "state" in data:
                state = int(data["state"][env_id].item())
                if state == DIRTY_STATE:
                    dirty_objects.append(name)

            if x > -0.4 and y < 0:
                detected_obj["table"][name] = pos

            if x < -0.6 and y > 0.2:
                detected_obj["trashcan"][name] = pos

            if x < -0.35 and y < 0:
                detected_obj["drying_zone"][name] = pos

            if x > 0.2 and y < 0:
                detected_obj["sink"][name] = pos

            if -0.4 < x < -0.12 and y > 0:
                detected_obj["food_storage"][name] = pos

            if x > -0.12 and y > 0:
                detected_obj["dish_storage"][name] = pos

        def verifier(new_obs: Dict) -> ToolResult:
            if all(not objects for objects in detected_obj.values()):
                reason = "There is no object detected."
            else:
                reason = "This is the position of existing objects:"

                for location, objects in detected_obj.items():
                    if not objects:
                        continue

                    objs = ", ".join(objects.keys())
                    reason += f" {location} contains {objs}."

            if not dirty_objects:
                reason += " There are no dirty objects."
            elif len(dirty_objects) == 1:
                reason += f" Dirty object is {dirty_objects[0]}."
            else:
                reason += f" Dirty objects are {', '.join(dirty_objects)}."

            return ToolResult(
                True,
                reason,
                context={
                    "objects_by_location": detected_obj,
                    "dirty_objects": dirty_objects,
                },
            )

        return ToolExecution(poses=["OK"], verifier=verifier)

    @register_tool(
        description="Pick up an object with the selected robot.",
        params_spec={"name": {"description": "Name of the object to pick up", "type": str}},
        errors=[
            ToolErrorSupport(AdvancedMaskItemsError, pre=True, post=False),
            ToolErrorSupport(GraspItemsFailureError, pre=True, post=False),
            ToolErrorSupport(OneShotToolFailureError, pre=True, post=False),
        ],
    )
    def take(self, obs: Observation, env_id: int, params: dict) -> ToolExecution:
        robot_name, tcp_key = self._robot_context(obs)

        if robot_name is None:
            return self._return_failed_tool("Unknown robot.")

        object_name = params["name"]

        if object_name not in obs.maniskill_obs["extra"]:
            return self._return_failed_tool(f"{object_name} was not detected.")

        location = self._object_location(obs, env_id, object_name,  ADVANCED_LOCATIONS)

        if robot_name == DISH_ROBOT:
            if object_name not in dishware:
                return self._return_failed_tool("The dish robot can only take dishware.")

            if location not in ("table", "sink", "drying_zone"):
                return self._return_failed_tool(f"The dish robot cannot reach {location}.")

        elif robot_name == TABLE_ROBOT:
            allowed_sources = (
                "table",
                "drying_zone",
                "food_storage",
                "dish_storage",
            )

            if location not in allowed_sources:
                return self._return_failed_tool(f"The table robot cannot reach {location}.")

        tool_execution = self._take_object(
            obs=obs,
            env_id=env_id,
            object_name=object_name,
            tcp_key=tcp_key,
            robot_name=robot_name,
            final_pose=ROBOT_POSE.get(robot_name,None)
        )
        tool_execution.execution_group = EXECUTION_GROUP_BY_LOCATION.get(location)
        return tool_execution

    @register_tool(
        description="Place the held object in a target location.",
        params_spec={"target": { "description": "Target location", "type": str}},
        errors=[
            ToolErrorSupport(OneShotToolFailureError, pre=True, post=False),
        ],
    )
    def put(self, obs: Observation, env_id: int, params: dict,) -> ToolExecution:
        robot_name, tcp_key = self._robot_context(obs)

        if robot_name is None:
            return self._return_failed_tool("Unknown robot.")

        target = params["target"]
        held_object = self._held_object(obs, env_id, tcp_key)

        if held_object is None:
            return self._return_failed_tool("The gripper is empty.")

        state = int(obs.maniskill_obs["extra"][held_object]["state"][env_id].item())

        if robot_name == DISH_ROBOT:

            if target not in ("sink", "drying_zone"):
                return self._return_failed_tool(
                    "The dish robot can only put objects in he sink or drying zone."
                )

            if target == "sink" and state != DIRTY_STATE:
                return self._return_failed_tool(f"{held_object} is already clean.")

            if target == "drying_zone" and state != CLEAN_STATE:
                return self._return_failed_tool(f"{held_object} must be cleaned before drying.")

        elif robot_name == TABLE_ROBOT:
            allowed_targets = (
                "table",
                "food_storage",
                "dish_storage",
                "trashcan",
            )

            if target not in allowed_targets:
                return self._return_failed_tool(f"The table robot cannot put objects in {target}.")

            if target == "trashcan":
                if held_object not in food or state != DIRTY_STATE:
                    return self._return_failed_tool("Only dirty food can be thrown away.")

            if target == "food_storage":
                if held_object not in food or state != CLEAN_STATE:
                    return self._return_failed_tool("Only clean food can be put in food storage.")

            if target == "dish_storage":
                if held_object not in dishware or state != CLEAN_STATE:
                    return self._return_failed_tool("Only clean dishware can be stored.")

        tool_execution = self._put_object(
            obs=obs,
            env_id=env_id,
            target=target,
            tcp_key=tcp_key,
            robot_name=robot_name,
            container_targets=("trashcan",),
            final_pose=ROBOT_POSE[robot_name],
        )
        tool_execution.execution_group = EXECUTION_GROUP_BY_LOCATION.get(target)
        return tool_execution

    @register_tool(
        description="Clean dirty dishware located in the sink.",
        params_spec={"name": { "description": "Name of the dishware to clean", "type": str}},
        execution_group="drying_zone",
        errors=[
            ToolErrorSupport(OneShotToolFailureError, pre=True, post=False),
        ],
    )
    def clean(self, obs: Observation, env_id: int, params: dict,) -> ToolExecution:
        robot_name, tcp_key = self._robot_context(obs)
        object_name = params["name"]
        extra = obs.maniskill_obs["extra"]

        if robot_name != DISH_ROBOT:
            return self._return_failed_tool("Only the dish robot can clean dishware.")

        if object_name not in dishware or object_name not in extra:
            return self._return_failed_tool(f"{object_name} is not known dishware.")

        location = self._object_location(obs, env_id, object_name, ["sink"])

        if location != "sink":
            return self._return_failed_tool(f"{object_name} must be in the sink before cleaning.")

        state = int(extra[object_name]["state"][env_id].item())

        if state == CLEAN_STATE:
            return self._return_failed_tool(f"{object_name} is already clean.")

        if self._held_object(obs, env_id, tcp_key) is not None:
            return self._return_failed_tool("The gripper is not empty.")

        target_position = self._target_position(
            obs,
            env_id,
            "drying_zone",
            moving_object=object_name,
        )

        if target_position is None:
            return self._return_failed_tool("There is no free space in drying_zone.")

        object_pose = extra[object_name]["pose"][env_id]
        poses = compute_grasp_trajectory(
            self.get_agent(robot_name),
            object_pose.cpu().numpy(),
            add_seuil=0.2,
        )
        poses.extend(
            compute_drop_trajectory(
                self.get_agent(robot_name),
                drop_pose=target_position,
                drop_seuil=0.03,
                approach_seuil=0.2,
                final_pose=ROBOT_POSE[robot_name],
            )
        )

        def verifier(new_obs: Dict) -> ToolResult:
            new_extra = new_obs["extra"]
            new_object_pose = new_extra[object_name]["pose"][env_id]

            if is_object_in_gripper(
                new_extra[tcp_key]["pose"][env_id],
                new_object_pose,
            ):
                return ToolResult(False, reason=f"{object_name} is still in the gripper.")

            if not is_object_inside_target(
                new_object_pose,
                new_extra["drying_zone"]["pose"][env_id],
                thresh=0.2,
            ):
                return ToolResult(False, reason=f"{object_name} is not in drying_zone.")

            return ToolResult(
                True,
                reason=f"{object_name} is now clean and in drying_zone.",
                logs=Log(
                    content={
                        "object": object_name,
                        "state": CLEAN_STATE,
                        "target": "drying_zone",
                        "robot": robot_name,
                    }
                ),
                state_updates=[
                    EnvStateUpdate(
                        path=("magma_extra_state", object_name),
                        value=CLEAN_STATE,
                    )
                ],
            )

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            context={"target_name": object_name},
            allowed_moving_actors=[object_name],
        )

    @register_tool(
        description="Wipe the empty table with the sponge.",
        params_spec={},
        errors=[
            ToolErrorSupport(OneShotToolFailureError, pre=True, post=False),
        ],
    )
    def wipe(self, obs: Observation, env_id: int, params: dict) -> ToolExecution:
        robot_name, tcp_key = self._robot_context(obs)
        extra = obs.maniskill_obs["extra"]

        if robot_name != TABLE_ROBOT:
            return self._return_failed_tool("Only the table robot can wipe the table.")

        held_object = self._held_object(obs, env_id, tcp_key)

        if held_object != "sponge":
            return self._return_failed_tool("The table robot must hold the sponge.")

        table_pose = extra["table"]["pose"][env_id]

        for object_name in [*food, *dishware]:
            if object_name not in extra:
                continue

            if is_object_inside_target(extra[object_name]["pose"][env_id], table_pose,thresh=0.2):
                return self._return_failed_tool("The table must be empty before wiping.")

        poses = compute_swipe_trajectory(
            self.get_agent(robot_name),
            self.table_center,
            thresh=0.12,
            z_offset=0.01,
        )

        def verifier(new_obs: Dict) -> ToolResult:
            new_extra = new_obs["extra"]

            if not is_object_in_gripper(
                new_extra[tcp_key]["pose"][env_id],
                new_extra["sponge"]["pose"][env_id],
                threshold=0.02,
            ):
                return ToolResult(False, reason="The sponge was dropped while wiping.")

            return ToolResult(
                True,
                reason="The table was wiped successfully.",
                logs=Log(
                    content={
                        "object": "table",
                        "robot": robot_name,
                    }
                ),
            )

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            allowed_moving_actors=["sponge"],
        )

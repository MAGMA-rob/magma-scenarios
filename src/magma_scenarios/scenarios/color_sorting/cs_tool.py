# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.simulation.tools import BaseToolsAPI, register_tool
from magma_core.simulation.data_structures import ToolExecution, ToolResult, Observation, Log
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_core.simulation.utils.gripper_utils import find_object_in_gripper, is_object_in_gripper

from magma_scenarios.utils import compute_grasp_trajectory, compute_drop_trajectory
from magma_scenarios.templates.tools import PlacementGrid
import torch
from typing import Dict, List

class ColorDetectionTools(BaseToolsAPI):

    placement_grid = PlacementGrid(
        name="color_sorting_tray",
        rows=3,
        columns=3,
        cell_spacing=0.1,
    )

    def _get_scene_colors(self, obs: Observation) -> List[str]:
        colors = []

        for name in obs.maniskill_obs["extra"].keys():
            if name.endswith("_tray"):
                colors.append(name[:-len("_tray")])

        return colors


    def _get_tray_pose_names(self, obs: Observation) -> List[str]:
        return [
            name
            for name in obs.maniskill_obs["extra"].keys()
            if name.endswith("_tray")
        ]

 
    @register_tool(
        description="List visible cubes and their locations.",
        params_spec={},
        is_detection=True,
    )
    def get_object_state(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]

        detected_obj = {"table": {}}

        for name in extra.keys():
            if name.endswith("_tray"):
                color = name[:-len("_tray")]
                detected_obj[f"{color}_tray"] = {}

        for name, pose in extra.items():
            if "cube" not in name:
                continue

            cube_pos = pose[env_id]
            cube_color = name.split("_cube_")[0]
            found = False

            for location in detected_obj.keys():
                if location == "table":
                    continue

                color = location[:-len("_tray")]
                tray_pose = extra[f"{color}_tray"][env_id]

                if is_object_inside_target(cube_pos, tray_pose, 0.3):
                    detected_obj[location][cube_color] = (
                        detected_obj[location].get(cube_color, 0) + 1
                    )
                    found = True
                    break

            if not found:
                detected_obj["table"][cube_color] = (
                    detected_obj["table"].get(cube_color, 0) + 1
                )

        def verifier(new_obs: Dict) -> ToolResult:
            if all(not v for v in detected_obj.values()):
                return ToolResult(
                    True,
                    "there is no cube detected",
                    context=detected_obj,
                    logs=Log(""),
                )

            s = "this is the position of existing cubes:"

            for location, objects in detected_obj.items():
                if len(objects) == 0:
                    continue

                objs = ", ".join(
                    f"{count} {color} cube" if count == 1 else f"{count} {color} cubes"
                    for color, count in objects.items()
                )

                s += f" {location} contains {objs}."

            return ToolResult(True, s, context=detected_obj, logs=Log(""))

        return ToolExecution(poses=["OK"], verifier=verifier)

    @register_tool(
            description=(
                "Take a cube of a given color from a location."
            ),
            params_spec={
                "obj_color": {
                    "description": "The color of the cube to take.",
                    "type": str,
                },
                "source": {
                    "description": (
                        "The location containing the cube. "
                        "Use table or a tray name such as white_tray."
                    ),
                    "type": str,
                },
            },
    )
    def take(self, obs: Observation, env_id, params : Dict) -> ToolExecution:
        poses = []
        extra = obs.maniskill_obs["extra"]


        obj_color = params.get("obj_color", None)
        source = params.get("source", None)


        if source != "table" and source not in extra:
            return ToolExecution(poses=[], verifier=None, reason=f"There is no location named {source}.")

        cube_poses = {
            object_name: object_pose[env_id][:3]
            for object_name, object_pose in extra.items()
            if "_cube_" in object_name
        }
        held_object = None
        if cube_poses:
            held_object = find_object_in_gripper(
                extra["agent_tcp"][env_id][:3],
                cube_poses,
            )
        if held_object is not None and held_object.startswith(f"{obj_color}_cube_"):
            return ToolExecution(
                poses=[],
                verifier=None,
                reason=f"The object {held_object} is already in the gripper.",
            )

        selected_object = None

        for object_name, object_pose in extra.items():
            if not object_name.startswith(f"{obj_color}_cube_"):
                continue

            if source == "table":
                object_is_in_tray = any(
                    is_object_inside_target(
                        object_pose[env_id],
                        extra[tray_name][env_id],
                        thresh=0.3,
                        keep_tensor=False,
                    )
                    for tray_name in self._get_tray_pose_names(obs)
                )

                if not object_is_in_tray:
                    selected_object = object_name
                    break

            else:
                if is_object_inside_target(
                    object_pose[env_id],
                    extra[source][env_id],
                    thresh=0.3,
                    keep_tensor=False,
                ):
                    selected_object = object_name
                    break

        if selected_object is None:
            return ToolExecution(poses=[], verifier=None, reason=(f"There is no {obj_color} cube instance in {source}."))

        obj_pose = obs.maniskill_obs["extra"].get(selected_object)

        poses = compute_grasp_trajectory(self.get_agent(), obj_pose[env_id].cpu().numpy())
        
        def verifier(new_obs: Dict) -> ToolResult:
            ok = False
            obj_pos = new_obs["extra"][selected_object]
            ok = is_object_in_gripper(
                new_obs["extra"]["agent_tcp"][env_id],
                obj_pos[env_id],
                threshold=0.02,
            )
            if ok:
                reason = f"You have {selected_object} in your gripper"
            else:
                reason = f"Failed to grasp {obj_color} cube due to planning error."
            return ToolResult(ok,reason)

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            reason="",
            context={"target_name": selected_object, "obj_color": obj_color, "source": source},
            allowed_moving_actors=[selected_object],
        )

    @register_tool(
            description="Put the held cube in a tray.",
            params_spec={
                "target": {"description": "Name of the destination tray.", "type": str}
            }
    )
    def put(self, obs : Observation, env_id : int, params: Dict) -> ToolExecution:
        r = ""

        obj_in_gripper = None
        poses = []

        target = params["target"]

        if not target in obs.maniskill_obs["extra"]:
            r = f"No {target} in the scene."
            return ToolExecution(poses=poses, verifier=None, reason=r)

        reduced_obs = {k: v[env_id][:3] for k, v in obs.maniskill_obs["extra"].items()}
        agent_tcp_pos = reduced_obs.pop("agent_tcp", None)[:3]
        obj_in_gripper = find_object_in_gripper(
            agent_tcp_pos,
            reduced_obs
        )
        if obj_in_gripper is None:
            return ToolExecution(poses=[], verifier=None, reason="There is no cube in the gripper.")

        cube_poses = {
            name: pose[env_id]
            for name, pose in obs.maniskill_obs["extra"].items()
            if "cube" in name
        }
        target_cell = self.placement_grid.allocate(
            center_pose=obs.maniskill_obs["extra"][target][env_id],
            object_poses=cube_poses,
            batch_context=obs.tool_batch_context,
            owner=obj_in_gripper,
            reservation_namespace=f"color_sorting:{target}",
            excluded_objects={obj_in_gripper},
        )

        if target_cell is None:
            r = f"The {target} has no free cell."
        else:
            target_pose = obs.maniskill_obs["extra"][target][env_id].clone()
            target_pose[:3] = target_cell.world_position
            target_pose = target_pose.cpu().numpy()
            poses = compute_drop_trajectory(
                self.get_agent(),
                drop_pose=target_pose,
                approach_seuil=0.13,
                drop_seuil=0.04,
            )
        
        def verifier(new_obs: Dict) -> ToolResult:
            obj_pose = new_obs["extra"][obj_in_gripper][env_id]
            if is_object_in_gripper(new_obs["extra"]["agent_tcp"][env_id], obj_pose):
                return ToolResult(False, reason=f"The object is still in the gripper")
            if not is_object_inside_target(obj_pose, new_obs["extra"][target][env_id], 0.3):
                return ToolResult(False,reason=f"The object is not in the tray and not in the gripper")
            return ToolResult(True, reason=f"Successfuly sent {obj_in_gripper} to {target}")

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            reason=r,
            allowed_moving_actors=[obj_in_gripper],
        )

# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tools import BaseToolsAPI, register_tool
from magma_core.base.data_structures import ToolErrorSupport, ToolExecution, ToolResult, Observation, Trajectory
from magma_core.utils.env_utils import is_object_inside_target
from magma_core.utils.gripper_utils import is_object_in_gripper, find_object_in_gripper
from magma_core.base.data_structures import Log

from magma_scenarios.utils import compute_grasp_trajectory, compute_drop_trajectory

from typing import Dict, List, Optional
import sapien, torch

# "[{\"name\": \"move_object_to_location\", \"description\": \"Depose the object currently inside the gripper to the specified target location.\", \"parameters\": {\"drop_zone\": {\"description\": \"the name of the target location.\", \"type\": \"str\"}}}, 
# {\"name\": \"grab_specific_object\", \"description\": \"Grasp the object corresponding to item_name.\", \"parameters\": {\"item_name\": {\"description\": \"the name of the object to grasp.\", \"type\": \"str\"}}}]", 
# {\"name\": \"launch_cycle\", \"description\": \"Launch the default sorting cycle. It sort only objects provided as keys in object_areas_mapping parameter. The cycle continue until there is no more objects since 5 minutes.\", \"parameters\": {\"object_areas_mapping\": {\"description\": \"Keys represent the name of object to detect, associated value correspond to the target_location name.\", \"type\": \"Dict\"}}}
# "long_memory": "Memory:\n- You must sort objects by grasping them and put them in their corresponding target store location.\n- I am still waiting to confirm a target location for clipboard\n- Default object target location : fork -> area1, clipboard -> area2, valve -> area3, paper_sheet -> area4, laptop -> area5, .", 
# "short_memory": "[{\"user\": \"Choose a clipboard and put it where it belongs.\", \"model\": \"I need to confirm the target location for the clipboard first. Could you please specify where the clipboard should be placed?\"}]",
#  "answers": "{\"think\": \"The user asked to use the default target location for the clipboard. The default location for clipboard is area2. I need to grasp the clipboard first. Then, I can put it in area2. I do not need to wait for a confirmation for the target location anymore. 
# The action is to take the clipboard using grab_specific_object function. I need to remember to depose it in area2 after the take finished.\", \"say\": \"Okay, I'll use the default target location for the clipboard.\", 
# \"action\": {\"default_system\": {\"name\": \"grab_specific_object\", \"arguments\": {\"item_name\": \"clipboard\"}}}}", "attributes": "{\"objects_name\": [\"fork\", \"clipboard\", \"valve\", \"paper_sheet\", \"laptop\"], \"targets_name\": [\"area1\", \"area2\", \"area3\", \"area4\", \"area5\"], \"known_robots\": [\"default_system\"]}"},

class WarehouseSortingTool(BaseToolsAPI):
    CYCLE_SORTED_DISTANCE_THRESHOLD = 0.1
    CYCLE_GRASP_APPROACH_Z = 0.14
    CYCLE_TRANSFER_Z = 0.42
    CYCLE_DROP_Z = 0.18
    CYCLE_DROP_APPROACH_Z = 0.36
    CYCLE_STAGING_X = -0.15
    CYCLE_STAGING_Y = 0.0

    def _make_top_down_pose(self, p) -> sapien.Pose:
        return sapien.Pose(p=p, q=[0, 1, 0, 0])

    def _cycle_staging_pose(self) -> sapien.Pose:
        return self._make_top_down_pose([
            self.CYCLE_STAGING_X,
            self.CYCLE_STAGING_Y,
            self.CYCLE_TRANSFER_Z,
        ])

    def _is_cycle_object_sorted(
            self,
            obs_extra: Dict,
            env_id: int,
            obj_name: str,
            target_name: str,
        ) -> bool:
        return torch.norm(
            obs_extra[obj_name][env_id][:2] - obs_extra[target_name][env_id][:2]
        ) < self.CYCLE_SORTED_DISTANCE_THRESHOLD

    def _find_held_cycle_object(
            self,
            obs_extra: Dict,
            env_id: int,
            obj_to_sort: List[str],
        ) -> Optional[str]:
        reduced_obs = {
            k: v[env_id][:3]
            for k, v in obs_extra.items()
            if k == "agent_tcp" or k in obj_to_sort
        }
        agent_tcp_pos = reduced_obs.pop("agent_tcp", None)
        if agent_tcp_pos is None:
            return None
        return find_object_in_gripper(agent_tcp_pos, reduced_obs)

    def _compute_cycle_drop_trajectory(
            self,
            obs_extra: Dict,
            env_id: int,
            target_name: str,
        ) -> Trajectory:
        target_pose = obs_extra[target_name][env_id].cpu().numpy()
        drop_pose = self._make_top_down_pose([
            target_pose[0],
            target_pose[1],
            self.CYCLE_DROP_Z,
        ])
        drop_approach_pose = self._make_top_down_pose([
            target_pose[0],
            target_pose[1],
            self.CYCLE_DROP_APPROACH_Z,
        ])
        return compute_drop_trajectory(
            self.get_agent(),
            drop_pose=drop_pose,
            approach_pose=drop_approach_pose,
            final_pose=self._cycle_staging_pose(),
        )

    def _compute_cycle_grasp_drop_trajectory(
            self,
            obs_extra: Dict,
            env_id: int,
            obj_name: str,
            target_name: str,
        ) -> Trajectory:
        obj_pose = obs_extra[obj_name][env_id].cpu().numpy()
        target_pose = obs_extra[target_name][env_id].cpu().numpy()

        grasp_approach_pose = self._make_top_down_pose([
            obj_pose[0],
            obj_pose[1],
            self.CYCLE_GRASP_APPROACH_Z,
        ])
        lift_pose = self._make_top_down_pose([
            obj_pose[0],
            obj_pose[1],
            self.CYCLE_TRANSFER_Z,
        ])
        transfer_pose = self._make_top_down_pose([
            (obj_pose[0] + target_pose[0]) / 2,
            (obj_pose[1] + target_pose[1]) / 2,
            self.CYCLE_TRANSFER_Z,
        ])

        poses = compute_grasp_trajectory(
            self.get_agent(),
            obj_pose,
            move_pos=grasp_approach_pose,
        )
        poses.extend([lift_pose, transfer_pose])
        poses.extend(self._compute_cycle_drop_trajectory(obs_extra, env_id, target_name))
        return poses

    def _compute_next_cycle_trajectory(
            self,
            obs_extra: Dict,
            env_id: int,
            obj_to_sort: List[str],
            assignment: Dict[str, str],
        ) -> Trajectory:
        held_obj = self._find_held_cycle_object(obs_extra, env_id, obj_to_sort)
        if held_obj in obj_to_sort:
            return self._compute_cycle_drop_trajectory(
                obs_extra,
                env_id,
                assignment[held_obj],
            )

        for obj_name in obj_to_sort:
            if self._is_cycle_object_sorted(
                obs_extra,
                env_id,
                obj_name,
                assignment[obj_name],
            ):
                continue

            return self._compute_cycle_grasp_drop_trajectory(
                obs_extra,
                env_id,
                obj_name,
                assignment[obj_name],
            )
        return []

    def take_obj(self, obs : Observation, env_id, params : Dict) -> ToolExecution:
        poses = []
        name_obj = None
        r = ""

        name_obj = params.get("obj", None)

        for obj_name, obj_pos in obs.maniskill_obs["extra"].items():
            if name_obj in obj_name:
                poses = compute_grasp_trajectory(self.get_agent(),obj_pos[env_id].cpu().numpy())
                break
        if not poses:
            r=f"No objects names corresponding to {name_obj}. You must pass the name of the object to take."

        # define verifier inline
        def verifier(new_obs: Dict) -> ToolResult:
            # e.g. check if gripper is holding the right object
            reason=f"No object with {name_obj} name was found. You must pass the name of the object to take."
            ok = False
            for obj_name, obj_pos in new_obs["extra"].items():
                if name_obj in obj_name:
                    ok = is_object_in_gripper(new_obs["extra"]["agent_tcp"][env_id], obj_pos[env_id])   
                    if ok:
                        reason = f"You have a {name_obj} object in your gripper"
                        break
                    else:
                        reason = f"Failed to grasp the {name_obj} object. You can retry."
            return ToolResult(ok, reason)
        return ToolExecution(poses=poses, verifier=verifier, reason=r)
    

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

        return ToolExecution(poses=poses, verifier=verifier, reason=r)
    
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



class WithManufacturingOrder(WarehouseSortingTool):

    @register_tool(
            description="Launch a default cycle to sort all objects to their assigned area.",
            params_spec={
                "assignment": {"description": "Dictionary of the object to sort as dictionary keys with their corresponding area.", "type": dict},
                "manu_order": {"description": "The Manufacturing Order associated with this cycle", "type": str}
            },
    )
    def launch_cycle(self, obs : Observation, env_id : int, params: Dict) -> ToolExecution:
        obj_to_sort, manu_order, assignment = [], "", {}
        task_attributes = obs.task_attributes

        def verifier(new_obs: Dict) -> ToolResult:
            held_obj = self._find_held_cycle_object(new_obs["extra"], env_id, obj_to_sort)
            if held_obj in obj_to_sort:
                return ToolResult(False, f"{held_obj} is still in the gripper. You can retry.")

            for obj_name in obj_to_sort:
                if not self._is_cycle_object_sorted(
                    new_obs["extra"],
                    env_id,
                    obj_name,
                    assignment[obj_name],
                ):
                    return ToolResult(
                        False,
                        "Cycle did not finish. You can retry.",
                    )

            s = ', '.join(f'{obj} to {ass}' for obj, ass in assignment.items())
            return ToolResult(True, f"All objects has been sorted : {s}", logs=Log(content=manu_order))

        assignment = params.get("assignment", "none")
        if assignment is None:
            return ToolExecution([],verifier=verifier,reason="Assignment parameter is invalid !")
        
        invalid_objects = []
        for key in assignment.keys():
            if (key not in task_attributes["objects"]):
                invalid_objects.append(key)

        if len(invalid_objects) > 0:
            return ToolExecution([],verifier=verifier,reason=f"There are no manipulable {str.join(', ', invalid_objects)} object(s).")

        invalid_areas = []
        for value in assignment.values():
            if (value not in task_attributes["target_areas"]):
                invalid_areas.append(value)

        if len(invalid_areas) == 1:
            return ToolExecution([],verifier=verifier,reason=f"This area {invalid_areas[0]} is invalid. Please use only known target.")
        elif (len(invalid_areas) > 1):
            return ToolExecution([],verifier=verifier,reason=f"These areas {str.join(', ', invalid_areas)} are invalid. Please use only known target.")
        
        manu_order = params['manu_order']

        for obj_name, target in assignment.items():
            if not self._is_cycle_object_sorted(obs.maniskill_obs["extra"], env_id, obj_name, target):
                obj_to_sort.append(obj_name)

        if not obj_to_sort:
            return ToolExecution([],verifier=verifier,reason=f"All objects has already been sorted.")

        cpt, cpt_max = 0, len(obj_to_sort) * 2 + 2

        def redo(new_obs: Dict) -> Trajectory:
            nonlocal cpt
            cpt +=1
            if cpt > cpt_max:
                return []
            return self._compute_next_cycle_trajectory(
                new_obs["extra"],
                env_id,
                obj_to_sort,
                assignment,
            )
        p = redo(obs.maniskill_obs)
        return ToolExecution(poses=p,verifier=verifier,redo=redo)
    
    @register_tool(
            description="Add a new location to the known area the robot can use to store objects",
            params_spec={
                "location_name": {"description": "The name of the new location. Must not already exist.", "type": str}
            }
    )
    def add_new_location(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        return super().add_new_location(obs, env_id, params)

    @register_tool(
            description="Remove an existing location from the known area.",
            params_spec={
                "location_name": {"description": "The name of the location to remove. Must exist.", "type": str}
            }
    )
    def remove_location(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        return super().remove_location(obs, env_id, params)
    

class WithoutManufacturingOrder(WarehouseSortingTool):

    @register_tool(
            description="Take an object from the environment.",
            params_spec={
                "obj": {"description": "The name of the object to take in the gripper.", "type": str},
            }
    )
    def take_obj(self, obs, env_id, params: Dict) -> ToolExecution:
        return super().take_obj(obs, env_id, params)
    
    @register_tool(
            description="Put the held object in an area",
            params_spec={
                "target": {"description": "Move the robot gripper upper the area and drop the current object.", "type": str}
            }
    )
    def depose(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        return super().depose(obs, env_id, params)

    @register_tool(
            description="Add a new location to the known area the robot can use to store objects",
            params_spec={
                "location_name": {"description": "The name of the new location. Must not already exist.", "type": str}
            }
    )
    def add_new_location(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        return super().add_new_location(obs, env_id, params)

    @register_tool(
            description="Remove an existing location from the known area.",
            params_spec={
                "location_name": {"description": "The name of the location to remove. Must exist.", "type": str}
            }
    )
    def remove_location(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        return super().remove_location(obs, env_id, params)

    @register_tool(
            description="Launch a default cycle to sort all objects to their assigned area.",
            params_spec={
                "assignment": {"description": "Dictionary of the object to sort as dictionary keys with their corresponding area.", "type": dict},
            },
    )
    def launch_cycle(self, obs : Observation, env_id : int, params: Dict) -> ToolExecution:
        obj_to_sort, assignment = [], {}
        task_attributes = obs.task_attributes

        def verifier(new_obs: Dict) -> ToolResult:
            held_obj = self._find_held_cycle_object(new_obs["extra"], env_id, obj_to_sort)
            if held_obj in obj_to_sort:
                return ToolResult(False, f"{held_obj} is still in the gripper. You can retry.")

            for obj_name in obj_to_sort:
                if not self._is_cycle_object_sorted(
                    new_obs["extra"],
                    env_id,
                    obj_name,
                    assignment[obj_name],
                ):
                    return ToolResult(
                        False,
                        "Cycle did not finish. You can retry.",
                    )

            s = ', '.join(f'{obj} to {ass}' for obj, ass in assignment.items())
            return ToolResult(True, f"All objects has been sorted : {s}",logs=Log(""))

        assignment = params.get("assignment", "none")
        if assignment is None:
            return ToolExecution([],verifier=verifier,reason="Assignment parameter is invalid !")
        
        invalid_objects = []
        for key in assignment.keys():
            if (key not in task_attributes["objects"]):
                invalid_objects.append(key)

        if len(invalid_objects) > 0:
            return ToolExecution([],verifier=verifier,reason=f"There are no manipulable {', '.join(map(str, invalid_objects))} object(s).")

        invalid_areas = []
        for value in assignment.values():
            if (value not in task_attributes["target_areas"]):
                invalid_areas.append(value)

        if len(invalid_areas) == 1:
            return ToolExecution([],verifier=verifier,reason=f"This area {invalid_areas[0]} is invalid. Please use only known target.")
        elif (len(invalid_areas) > 1):
            return ToolExecution([],verifier=verifier,reason=f"These areas {', '.join(map(str, invalid_areas))} are invalid. Please use only known target.")

        for obj_name, target in assignment.items():
            if not self._is_cycle_object_sorted(obs.maniskill_obs["extra"], env_id, obj_name, target):
                obj_to_sort.append(obj_name)

        if not obj_to_sort:
            return ToolExecution([],verifier=verifier,reason=f"All objects has already been sorted.")

        cpt, cpt_max = 0, len(obj_to_sort) * 2 + 2

        def redo(new_obs: Dict) -> Trajectory:
            nonlocal cpt
            cpt +=1
            if cpt > cpt_max:
                return []
            return self._compute_next_cycle_trajectory(
                new_obs["extra"],
                env_id,
                obj_to_sort,
                assignment,
            )
        p = redo(obs.maniskill_obs)
        return ToolExecution(poses=p,verifier=verifier,redo=redo)

from typing import Dict, List, Optional
import sapien
import torch
from magma_scenarios.utils import compute_grasp_drop_trajectory
from magma_core.simulation.tools import BaseToolsAPI, register_tool
from magma_core.simulation.data_structures import (
    ToolErrorSupport,
    ToolExecution,
    ToolResult,
    Observation,
    EnvStateUpdate,
)
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_core.simulation.data_structures import Log
from magma_scenarios.templates.errors import (
    OneShotToolFailureError,
    RequestedObjectGraspFailureError,
)
from magma_scenarios.templates.tools import PlacementGrid

class HallSortingTool(BaseToolsAPI):
    """
    Tool API for the four-hall sorting scenario.
    """

    tray_capacity = 4
    detect_radius = 0.9
    object_height = 0.02
    GAP = 0.15
    placement_grid = PlacementGrid(
        name="hall_sorting_support",
        rows=2,
        columns=2,
        cell_spacing=0.2,
    )

    table_positions = [
        [0.0, 2.0],    # top
        [-2.0, 0.0],   # left
        [2.0, 0.0],    # right
        [0.0, -2.0],   # bottom
    ]

    robot_positions = {
        "Hall1" : [-2.3, -0.3, 0],   # near left table
        "Hall2" : [-0.3, 1.7, 0],    # near top table
        "Hall3" : [1.7, -0.3, 0],    # near right table
        "Hall4" : [-0.3, -2.3, 0],   # near bottom table
        
    }


    def _selected_agent_name(self, obs):
        return obs.selected_robot_name

    def _selected_agent(self, obs):
        return self.get_agent(self._selected_agent_name(obs))

    def _selected_robot(self, obs):
        return self._selected_agent(obs).robot

    def _robot_idx(self, obs) -> int:
        robot_name = self._selected_agent_name(obs)
        return self.equivalence[robot_name]

    def _tray_name(self,obs) -> str:
        return f"tray-{self._robot_idx(obs)}"

    def _hall_names(self, extra: Dict) -> List[str]:
        return sorted([name for name in extra.keys() if name.startswith("Hall")])


    def _objects_on_tray(self, obs, env_id: int) -> List[str]:
        extra = obs.maniskill_obs["extra"]
        tray_name = self._tray_name(obs)

        if tray_name not in extra:
            return []

        tray_pos = extra[tray_name][env_id]
        objects = []

        for name, pose in extra.items():
            if not self._is_object(name):
                continue
            on_tray = is_object_inside_target(pose[env_id],tray_pos,0.2,keep_tensor=False)
            if on_tray :
                objects.append(name)

        return objects

    def _objects_near_robot(self, obs, env_id: int) -> List[str]:
        extra = obs.maniskill_obs["extra"]
        hall_name = self._current_hall(obs,env_id)
        hall_pose = extra[hall_name][env_id]

        objects = []

        for name, pose in extra.items():
            if not self._is_object(name):
                continue
            obj_pos = pose[env_id]
            near = is_object_inside_target(obj_pos, hall_pose, 0.2)
            if near:
                objects.append(name)

        return objects

    def _current_hall(self, obs, env_id: int) -> Optional[str]:
        extra = obs.maniskill_obs["extra"]
        tray_name = self._tray_name(obs)

        if tray_name not in extra:
            return None

        tray_pos = extra[tray_name][env_id][:3]

        best_hall = None
        best_dist = float("inf")

        for hall_name in self._hall_names(extra):
            hall_pos = extra[hall_name][env_id][:3]
            dist = torch.norm(tray_pos[:2] - hall_pos[:2]).item()
            if dist < best_dist:
                best_hall = hall_name
                best_dist = dist

        return best_hall

    def _is_object(self, name: str) -> bool:
        return (
            name.startswith("book_")
            or name.startswith("pen_")
            or name.startswith("backpack_")
            or name.startswith("package_")
        )
        
    def _object_poses(self, obs: Observation, env_id: int):
        extra = obs.maniskill_obs["extra"]
        return {
            name: pose[env_id]
            for name, pose in extra.items()
            if self._is_object(name)
        }

    def _allocate_cell(
        self,
        obs: Observation,
        env_id: int,
        support_name: str,
        object_name: str,
    ):
        cell = self.placement_grid.allocate(
            center_pose=obs.maniskill_obs["extra"][support_name][env_id],
            object_poses=self._object_poses(obs, env_id),
            batch_context=obs.tool_batch_context,
            owner=object_name,
            reservation_namespace=f"hall_sorting:{support_name}",
            excluded_objects=[object_name],
        )
        return cell

    @register_tool(
        description="List objects near the selected robot.",
        params_spec={},
        is_detection=True,
    )
    def detect(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        table_objects = self._objects_near_robot(obs, env_id)
        tray_objects = self._objects_on_tray(obs, env_id)
        current_hall = self._current_hall(obs, env_id)

        def verifier(new_obs: Dict) -> ToolResult:
            if len(table_objects) == 0:
                reason = "No object detected on the current table."
            else:
                reason = "Objects on the current table: " + ", ".join(table_objects) + "."

            if len(tray_objects) == 0:
                reason += " The tray is empty."
            else:
                reason += " The tray contains: " + ", ".join(tray_objects) + "."

            if current_hall is not None:
                reason += f" The robot is near {current_hall}."

            return ToolResult(
                True,
                reason=reason,
                context={
                    "table_objects": table_objects,
                    "tray_objects": tray_objects,
                    "current_hall": current_hall,
                },
            )

        return ToolExecution(poses=["OK"], verifier=verifier)

    @register_tool(
        description="Move the selected robot to the requested hall.",
        params_spec={
            "hall": {
                "description": "Target hall name, for example Hall1, Hall2, Hall3 or Hall4.",
                "type": str,
            }
        },
    )
    def move_to(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]
        hall = params["hall"]

        if hall not in extra:
            return ToolExecution([], verifier=None, reason=f"Unknown hall: {hall}.")

        tray_name = self._tray_name(obs)
        if tray_name not in extra:
            return ToolExecution([], verifier=None, reason=f"{tray_name} does not exist.")

        robot = self._selected_robot(obs)
        selected_robot_name = self._selected_agent_name(obs)
        real_robot_name = robot.name

        current_hall = self._current_hall(obs, env_id)

        if current_hall == hall:
            return ToolExecution([], verifier=None, reason=f"{selected_robot_name} is already in {hall}.")

        for robot_name, robot_idx in self.equivalence.items():
            if robot_name == selected_robot_name:
                continue

            other_tray_name = f"tray-{robot_idx}"

            if other_tray_name not in extra:
                continue

            other_tray_pos = extra[other_tray_name][env_id][:3]
            target_hall_pos = extra[hall][env_id][:3]

            if torch.norm(other_tray_pos[:2] - target_hall_pos[:2]).item() < 0.75:
                return ToolExecution([],verifier=None,reason=f"{hall} is already occupied by {robot_name}.")

        # Target robot base pose near the selected hall.
        target_robot_state = robot.get_state()[env_id].clone()

        target_robot_xyz = torch.tensor(self.robot_positions[hall],device=target_robot_state.device,dtype=target_robot_state.dtype)

        # Target tray pose relative to the robot base.
        target_tray_xyz = target_robot_xyz.clone()
        target_tray_xyz[1] += 0.55
        target_tray_xyz[2] = 0.02


        q = torch.tensor(
            [1, 0, 0, 0],
            device=target_robot_state.device,
            dtype=target_robot_state.dtype,
        )

        target_robot_state[0:3] = target_robot_xyz
        target_robot_state[3:7] = q
        target_robot_state[7:13] = 0

        old_tray_pose = extra[tray_name][env_id].clone()
        tray_delta = target_tray_xyz - old_tray_pose[:3]

        target_tray_state = torch.zeros(13,  device=robot.get_state().device, dtype=robot.get_state().dtype)
        target_tray_state[0:3] = target_tray_xyz
        target_tray_state[3:7] = q
        target_tray_state[7:13] = 0

        state_updates = [
            EnvStateUpdate(
                path=("articulations", real_robot_name),
                value=target_robot_state,
            ),
            EnvStateUpdate(
                path=("actors", tray_name),
                value=target_tray_state,
            ),
        ]

        tray_objects = self._objects_on_tray(obs, env_id)

        # Move tray contents with the tray.
        for object_name in tray_objects:
            old_obj_pose = extra[object_name][env_id].clone()

            target_obj_state = torch.zeros(13, device=old_obj_pose.device, dtype=old_obj_pose.dtype)
            target_obj_state[0:3] = old_obj_pose[:3] + tray_delta
            target_obj_state[3:7] = old_obj_pose[3:7]
            target_obj_state[7:13] = 0

            state_updates.append(
                EnvStateUpdate(
                    path=("actors", object_name),
                    value=target_obj_state,
                )
            )

        def verifier(new_obs: Dict) -> ToolResult:
            return ToolResult(
                True,
                reason=f"{selected_robot_name} moved to {hall} with {tray_name}.",
                state_updates=state_updates,
            )

        return ToolExecution(poses=["OK"],verifier=verifier)

    @register_tool(
        description="Load an object from the current table onto the selected robot's tray.",
        params_spec={
            "object": {
                "description": "Name of the object to pick from the table.",
                "type": str,
            }
        },
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
    def pick_to_tray(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]
        object_name = params["object"]

        if object_name not in extra:
            return ToolExecution([], verifier=None, reason=f"Unknown object: {object_name}.")

        tray_name = self._tray_name(obs)
        if tray_name not in extra:
            return ToolExecution([], verifier=None, reason=f"{tray_name} does not exist.")

        tray_objects = self._objects_on_tray(obs, env_id)
        if object_name in tray_objects:
            return ToolExecution([], verifier=None, reason=f"{object_name} is already on the tray.")

        if len(tray_objects) >= self.tray_capacity:
            return ToolExecution([], verifier=None, reason="The tray is full. It supports only 4 objects.")

        if object_name not in self._objects_near_robot(obs, env_id):
            return ToolExecution([], verifier=None, reason=f"{object_name} is not on the current hall table.")

        target_cell = self._allocate_cell(
            obs,
            env_id,
            support_name=tray_name,
            object_name=object_name,
        )
        if target_cell is None:
            return ToolExecution([], verifier=None, reason="No free cell on the tray.")

        robot_idx = self._robot_idx(obs)
        obj_pose = extra[object_name][env_id]

        drop_xyz = [
            float(target_cell.world_position[0]),
            float(target_cell.world_position[1]),
            float(target_cell.world_position[2] + 0.04),
        ]

        drop_q = [0, 1, 0, 0]

        drop_pose = sapien.Pose(
            p=drop_xyz,
            q=drop_q,
        )

        drop_approach_pose = sapien.Pose(
            p=[
                drop_xyz[0],
                drop_xyz[1],
                drop_xyz[2] + 0.10,
            ],
            q=drop_q,
        )

        poses = compute_grasp_drop_trajectory(
            self._selected_agent(obs),
            obj_pose,
            drop_pose,
            approach_seuil=0.1,
            drop_seuil=0.07,
            drop_approach_pose=drop_approach_pose,
        )

        def verifier(new_obs: Dict) -> ToolResult:
            new_extra = new_obs["extra"]
            if object_name not in new_extra or tray_name not in new_extra:
                return ToolResult(False, "Missing object or tray after action.")

            obj_pos = new_extra[object_name][env_id]
            tray_pos = new_extra[tray_name][env_id]

            if is_object_inside_target(obj_pos, tray_pos, thresh=0.25):
                return ToolResult(True, f"{object_name} is now on the tray.")

            return ToolResult(False, f"{object_name} is not on the tray. You can retry.")

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            robot_idx=robot_idx,
            context={"target_name": object_name},
            allowed_moving_actors=[object_name],
        )

    @register_tool(
        description="Unload an object from the selected robot's tray onto the current table.",
        params_spec={
            "object": {
                "description": "Name of the object to remove from the tray and place on the table.",
                "type": str,
            }
        },
        errors=[
            ToolErrorSupport(
                OneShotToolFailureError,
                pre=True,
                post=False,
            ),
        ],
    )
    def drop_from_tray(self, obs: Observation, env_id: int, params: Dict) -> ToolExecution:
        extra = obs.maniskill_obs["extra"]
        object_name = params["object"]

        if object_name not in extra:
            return ToolExecution([], verifier=None, reason=f"Unknown object: {object_name}.")

        tray_name = self._tray_name(obs)
        tray_objects = self._objects_on_tray(obs, env_id)

        if object_name not in tray_objects:
            return ToolExecution([], verifier=None, reason=f"{object_name} is not on the tray.")

        current_hall = self._current_hall(obs, env_id)
        if current_hall is None:
            return ToolExecution([], verifier=None, reason="The robot is not near any hall.")

        target_cell = self._allocate_cell(
            obs,
            env_id,
            support_name=current_hall,
            object_name=object_name,
        )
        if target_cell is None:
            return ToolExecution([], verifier=None, reason=f"No free place on {current_hall}.")

        robot_idx = self._robot_idx(obs)
        obj_pose = extra[object_name][env_id]
        drop_xyz = [
            float(target_cell.world_position[0]),
            float(target_cell.world_position[1]),
            0.04,
        ]

        drop_q = [0, 1, 0, 0]

        drop_pose = sapien.Pose(
            p=drop_xyz,
            q=drop_q,
        )
        
        final_pose = sapien.Pose(
                p=[0, 0, 0.35],
                q=[0,1,0,0]
            )

        poses = compute_grasp_drop_trajectory(
            self._selected_agent(obs),
            obj_pose,
            drop_pose,
            approach_seuil=0.1,
            drop_seuil=0.08,
            final_pose=final_pose
        )
        

        def verifier(new_obs: Dict) -> ToolResult:
            new_extra = new_obs["extra"]
            obj_pos = new_extra[object_name][env_id]
            hall_pos = new_extra[current_hall][env_id]
            tray_pos = new_extra[tray_name][env_id]

            if is_object_inside_target(obj_pos, tray_pos, thresh=0.25):
                return ToolResult(False, f"{object_name} is still on the tray.")

            if not is_object_inside_target(obj_pos, hall_pos, thresh=0.25):
                return ToolResult(False, f"{object_name} is not on {current_hall}.")

            return ToolResult(True, f"{object_name} was dropped on {current_hall}.",logs=Log(content={"object": object_name,"hall": current_hall}))

        return ToolExecution(
            poses=poses,
            verifier=verifier,
            robot_idx=robot_idx,
            allowed_moving_actors=[object_name],
        )

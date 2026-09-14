from typing import Dict
from magma_core.simulation.data_structures.tools import ToolErrorSupport
from magma_core.simulation.tools import BaseToolsAPI, register_tool
from magma_core.simulation.data_structures import ToolExecution, ToolResult, Observation, Log
from magma_core.simulation.utils.gripper_utils import find_object_in_gripper, is_object_in_gripper
import sapien
from .packaging_errors import MaskFoodError, GraspFoodFailureError
from magma_scenarios.templates.errors import OneShotToolFailureError
from magma_scenarios.utils import compute_drop_trajectory, compute_grasp_trajectory
from magma_core.simulation.utils.env_utils import is_object_inside_target
import torch
from .attributes import fruits, drinks, main_course
from magma_scenarios.templates.tools import PlacementGrid

class PackagingTool(BaseToolsAPI):

    table_gride_centre = [-0.1,-0.2,0]
    placement_grid = PlacementGrid(
        name="packaging_support",
        rows=3,
        columns=3,
        cell_spacing=0.11,
        selection_order=[8, 7, 6, 5, 4, 3, 2, 1, 0],
    )


    @register_tool(
        description="List the objects on the table and on the tray.",
        params_spec={},
        is_detection=True,
        errors = [
            ToolErrorSupport(MaskFoodError,pre = False, post= True)
        ]
    )
    def detect(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        
        extra = obs.maniskill_obs["extra"]
        detected_obj = {"table" : {},"tray" : {}}
        for name , pos in extra.items() :
            if name in ["agent_tcp","tray"]:
                continue
            y = pos[env_id][1]
            if y < 0 :
                detected_obj["table"][name] = pos[env_id][:3]
            else :
                detected_obj["tray"][name] = pos[env_id][:3]

        def verifier(new_obs: Dict) -> ToolResult:
            s = "this is the position of existing object : "

            if len(detected_obj["table"]) == 0 :
                s += "the table is empty"
            else :
                s += "the table contain " + ', '.join(f"{name}" for name in detected_obj["table"].keys())

            if len(detected_obj["tray"]) == 0 :
                s += ", and the tray is empty"
            else :
                s += ", and the tray contain " + ', '.join(f"{name}" for name in detected_obj["tray"].keys())

            return ToolResult(True,s,context = detected_obj)

        return ToolExecution(poses = ["OK"], verifier=verifier)


    @register_tool(
        description="Take an object from the table.",
        params_spec={"name" : {"description" : "Name of the object to take.", "type": str}},
        errors = [
            ToolErrorSupport(GraspFoodFailureError, pre=True, post=False),
            ToolErrorSupport(OneShotToolFailureError, pre=True, post=False),
        ]
    )
    def take(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        """go fetch an object by his name """

        name = params["name"]
        extra = obs.maniskill_obs["extra"]
        reduced_env = {obj : pos[env_id][:3] for obj, pos in extra.items() if obj in [*fruits, *drinks, *main_course]}

        agent_tcp_position = extra["agent_tcp"][env_id][:3]
        if find_object_in_gripper(agent_tcp_position, reduced_env) :
            return ToolExecution(poses=[],verifier=None,reason="The gripper is not empty.")

        if not name in obs.maniskill_obs["extra"]:
            return ToolExecution(
                poses=[],
                verifier=None,
                reason=f"{name} is not detected"
            )

        obj_pose = obs.maniskill_obs["extra"][name][env_id]

        poses = compute_grasp_trajectory(self.get_agent(),obj_pose.cpu().numpy())

        def verifier(new_obs: dict) -> ToolResult:
            "the object must be in the gripper"
            new_extra = new_obs["extra"]
            object = new_extra.get(name, None)
            if object is None:
                return ToolResult(False, f"the object {name} does not exist anymore ")
            if is_object_in_gripper(new_extra["agent_tcp"][env_id],
                                    object[env_id],
                                    threshold=0.02) :
                return ToolResult(True, f"you have {name} in your gripper")
            else :
                return ToolResult(
                    False, f"you failed to take the object {name}. you can retry"
                )
        return ToolExecution(
            poses,
            verifier=verifier,
            context={"target_name": name},
            allowed_moving_actors=[name],
        )


    @register_tool(
        description="Put the held object on the table or on the tray.",
        params_spec={"target" : {"description" : "Destination: table or tray.", "type" : str}}
    )
    def put(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        extra = obs.maniskill_obs["extra"]
        reduced_env = {obj : pose[env_id][:3] for obj, pose in extra.items()}
        agent_tcp_position = reduced_env.pop("agent_tcp")
        obj_in_gripper = find_object_in_gripper(agent_tcp_position, reduced_env)

        if obj_in_gripper is None:
            return ToolExecution(
                poses = [], verifier= None, reason = "The gripper is empty. Aborting."
            )

        target = params["target"]
        if target == "table":
            reference = extra[obj_in_gripper][env_id]
            support_pose = torch.tensor(
                [*self.table_gride_centre, 1.0, 0.0, 0.0, 0.0],
                device=reference.device,
                dtype=reference.dtype,
            )
        elif target == "tray":
            support_pose = extra["tray"][env_id]
        else:
            return ToolExecution(
                poses = [], verifier= None, reason = f"Unknow {target} used as target. Use only tray or table."
            )

        food_names = {*fruits, *drinks, *main_course}
        food_poses = {
            name: pose[env_id]
            for name, pose in extra.items()
            if name in food_names
        }
        target_cell = self.placement_grid.allocate(
            center_pose=support_pose,
            object_poses=food_poses,
            batch_context=obs.tool_batch_context,
            owner=obj_in_gripper,
            reservation_namespace=f"packaging:{target}",
            excluded_objects=[obj_in_gripper],
        )
        if target_cell is None :
            r = f"there is no place in the {target}"
            return ToolExecution(
                poses = [], verifier= None, reason = r
            )
        target_pose = target_cell.world_position.cpu().numpy()

        poses = [sapien.Pose(agent_tcp_position[:3].cpu().numpy() + (0, 0, 0.1), (0, 1, 0, 0))]
        poses.extend(compute_drop_trajectory(
            self.get_agent(),
            drop_pose = target_pose,
            drop_seuil = 0.03,
            approach_seuil = 0.1
        ))
        def verifier(new_obs: dict) -> ToolResult:
            new_extra = new_obs["extra"]
            obj_pose = new_extra[obj_in_gripper][env_id]
            if is_object_in_gripper(
                new_extra["agent_tcp"][env_id], obj_pose):
                return ToolResult(False, reason = "the food is still in the gripper")
            tensor_target_pose = torch.tensor(target_pose, device=obj_pose.device)
            if not is_object_inside_target(obj_pose, tensor_target_pose) :
                return ToolResult(False, reason=f"The {obj_in_gripper} is not in {target} and not in the gripper")
            return ToolResult(True,reason=f"Successfully put {obj_in_gripper} in the {target}")

        return ToolExecution(
            poses,
            verifier,
            allowed_moving_actors=[obj_in_gripper],
        )
            

    @register_tool(
        description="Validate the tray and give it to the user.",
        params_spec={}
    )
    def valid_plate(self, obs: Observation, env_id: int, params: dict)-> ToolExecution :
        "the plate must contain at least one food"
        "the plate must not have two food from the category"
        def verifier(new_obs: dict)-> ToolResult:
            extra = new_obs["extra"]
            tray_content = {"fruits" : [], "drinks": [], "main_course": []}
            for name, pose in extra.items() :
                if name in ["agent_tcp", "tray"] :
                    continue
                if is_object_inside_target(pose[env_id], extra["tray"][env_id],thresh=0.2) :
                    if name in fruits :
                        tray_content["fruits"].append(name)
                    if name in drinks :
                        tray_content["drinks"].append(name)
                    if name in main_course :
                        tray_content["main_course"].append(name)

            return ToolResult(True,f"the tray is ready ! ",logs = Log(content=tray_content))
        return ToolExecution(["OK"], verifier=verifier)

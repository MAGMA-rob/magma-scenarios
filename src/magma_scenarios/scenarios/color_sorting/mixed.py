# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# from ..base import ToolsExecutor, ToolExecution, ToolResult
# from magma_core.utils.env_utils import find_object_in_gripper, is_object_in_gripper, verify_parameters_dict, is_object_inside_target
# from .detection import SortColorWithDetection
# import sapien, torch
# from typing import List, Dict

# from importlib import resources
# import yaml

# class MixedSortColorWithDetection(SortColorWithDetection):
#     """
#     Executor for the sort color cube task linked to the environment : MixedColorSorting.
#     """

#     name : str = "Sorting Cube by Color with Detection"

#     def __init__(self, target_steps: int=7, acceptance_steps: int=1, **kwargs):
#         kwargs.setdefault("scenario", '')
#         with resources.files(__package__).joinpath("detection_randomization.yaml").open("r") as f:
#            config = yaml.safe_load(f)
#         kwargs.setdefault("randomized_config", config)
#         super().__init__(target_steps, acceptance_steps, **kwargs)

#         self.env_id = "MixedColorSorting"
#         self._registry = {
#             "get_objects_state": self._get_object_state,
#             "take_object_per_id": self._take_object_per_id,
#             "send_object_to_box": self._put_to_box
#         }

#     def _build_init_elements(self):
#         self.memory = [
#         "You are in charge of sorting object by color.",
#         "To clean the table, each object must be in the box of the same color"
#         ]
#         self.preserved_memory_indice = [0,1]
#         self.attributes = {}
#         self.env_options = {"nb_mixed":1}
#         self.instruction = "Clean me the table please"

    
#     def compute_partial_reward(self, node_id: int) -> float:
#         node_infos = self._get_node_infos(node_id)
#         nb_red = 0
#         nb_black = 0
#         for obj_name, obj_pose in node_infos["env_state"]["actors"].items():
#             if "red_cube" in obj_name:
#                 nb_red += int(is_object_inside_target(obj_pose,node_infos["env_state"]["actors"]["red_box"]))
#             elif "black_cube" in obj_name:
#                 nb_black += int(is_object_inside_target(obj_pose,node_infos["env_state"]["actors"]["black_box"]))
#         return (nb_red+nb_black)/3
    
#     def _get_object_state(self, env_state, env_id, params : Dict) -> ToolExecution:
#         out, r = verify_parameters_dict(params, {})

#         detected_obj = {"red_box":[], "black_box":[], "table":[]}

#         if not out:
#             return ToolExecution(poses=[],verifier=None,reason=r)

#         for obj_name, obj_pose in env_state['extra'].items():
#             if "cube" in obj_name:
#                 if is_object_inside_target(obj_pose[env_id],env_state["extra"]["red_box_pose"][env_id]):
#                     detected_obj["red_box"].append(obj_name)
#                 elif is_object_inside_target(obj_pose[env_id],env_state["extra"]["black_box_pose"][env_id]):
#                     detected_obj["black_box"].append(obj_name)
#                 else:
#                     detected_obj["table"].append(obj_name)

#         def verifier(new_env_state: Dict) -> ToolResult:
#             self._add_logs(env_id, "get_object")
#             s = "This is the position of existing objects: "

#             if len(detected_obj['red_box']) == 0:
#                 s += "red_box is empty. "
#             else:
#                 s += ",".join(detected_obj["red_box"]) + " are in the red_box. "

#             if len(detected_obj['black_box']) == 0:
#                 s += "red_box is empty. "
#             else:
#                 s += ",".join(detected_obj['black_box']) + " are in the black_box. "

#             if len(detected_obj['table']) > 0:
#                 s += ",".join(detected_obj['table']) + " are not sorted."

#             return ToolResult(True,s)

#         return ToolExecution(poses=["OK"], verifier=verifier)
    
#     def _take_object_per_id(self, env_state, env_id, params : Dict) -> ToolExecution:
#         poses = []
#         obj_name = None
#         out, r = verify_parameters_dict(params, {"name":str})

#         if out:
#             obj_name = params.get("name", None)
#             obj_pose = env_state["extra"].get(obj_name,None)
#             if obj_pose != None:
#                 poses = self._compute_grasp_trajectory(obj_pose[env_id].cpu().numpy(), add_seuil=0.3)
#             else:
#                 r=f"{obj_name} is not a known objects. You can use get_objects_state to see all detected objects."

#         # define verifier inline
#         def verifier(new_env_state: Dict) -> ToolResult:
#             ok = False
#             obj_pos = new_env_state["extra"][obj_name]
#             ok = is_object_in_gripper(new_env_state["extra"]["agent_tcp"][env_id], obj_pos[env_id])   
#             if ok:
#                 reason = f"You have {obj_name} in your gripper"
#             else:
#                 reason = f"Failed to grasp {obj_name} due to planning error."
#             return ToolResult(ok,reason)

#         return ToolExecution(poses=poses, verifier=verifier, reason=r)

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         N = env_state["extra"]["agent_tcp"].shape[0]
#         device = env_state["extra"]["agent_tcp"].device
#         nb_black = torch.zeros(N, device=device)
#         nb_red = torch.zeros(N, device=device)

#         for obj_name, obj_pose in env_state["extra"].items():
#             if "red_cube" in obj_name:
#                 nb_red += is_object_inside_target(obj_pose,env_state["extra"]["red_box_pose"]).int()
#             elif "black_cube" in obj_name:
#                 nb_black += is_object_inside_target(obj_pose,env_state["extra"]["black_box_pose"]).int()

#         out = torch.zeros(N, device=device)
#         out[(nb_black == 1) & (nb_red == 2)] = 1

#         return out
    
# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# from ..base import ToolExecution, ToolResult
# from .main import TrayPickingResearchTools
# from magma_core.utils import find_object_in_gripper, is_object_in_gripper, verify_parameters_dict, is_object_inside_target

# import sapien, torch
# from typing import List, Dict

# scenario : str = """Scenario :
# The robot is first asked to say how many product of type A it haves in front of it. Then it must sort all products in containers.

# Therefore, the robot must inform you that he have 3 product_A in front of him. If the robot say something else, you must output "STOP".
# If the robot correctly tells you that he have 3 product of type A in front of him. You must ask him to clean them all from the table.
# If the robot tell you something about a task in its memory or that it do not have access to the right function you must output "STOP" because the robot made an error.
# If the robot say that its task is done, you must output "STOP".
# If the robot do not know what to do, output "STOP".
# If the robot tell you about an error or that its task is done output "STOP".
# If the robot tell you about something not mentionned above, output "STOP".
# """

# class TrayPickingSequenceTools(TrayPickingResearchTools):
#     """
#     Executor for the tray picking task linked to the environment : tray_picking.
#     The idea is to give tools to the model to let him sort the object on the table and place then in multiple container according to a request.
#     Here it does not have access directly to object instances. It has only product type as attributes and it must detect them before taking them.
#     """

#     name : str = "Tray Picking"

#     def __init__(self, target_steps: int=8, acceptance_steps: int=1, **kwargs):
#         kwargs.setdefault("scenario", scenario)
#         super().__init__(target_steps, acceptance_steps, **kwargs)

#         self.env_id = "TrayPickingBase-v1"
#         self._registry = {
#             "take": self._take_cube,
#             "drop_to": self._put_to_target,
#             "get_unsorted_object" : self._get_object_per_type,
#             "get_sorted_object_from_containers" : self._get_sorted_object_from_container
#         }

#     def _build_init_elements(self):
#         self.memory = [
#         "You are in charge of sorting product into specfic containers.",
#         # "productA and productB must never be in two consecutive containers.",
#         "Each container must contains only one product at maximum."
#         ]
#         self.attributes = {"product_type": ["productA", "productB"], "containers_id": ["0","1","2","3","4","5"]}
#         self.instruction = "Hello, How many object of class product A do you see?"

#     def _get_object_per_type(self, env_state, env_id, params : Dict) -> ToolExecution:

#         out, r = verify_parameters_dict(params, {})

#         detected_obj = []

#         if not out:
#             return ToolExecution(poses=[],verifier=None,reason=r)

#         for obj_name, obj_pos in env_state['extra'].items():
#             if "ref" in obj_name:
#                 if obj_pos[env_id][0] > -0.2:
#                     detected_obj.append(self._name_sim_to_llm(obj_name))

#         def verifier(new_env_state: Dict) -> ToolResult:
#             self._add_logs(env_id, "get_object")
#             return ToolResult(True,f"Object on the table are {detected_obj}.")

#         return ToolExecution(poses=["OK"], verifier=verifier)

#     def _get_sorted_object_from_container(self, env_state, env_id, params : Dict) -> ToolExecution:

#         out, r = verify_parameters_dict(params, {})

#         obj_per_containers = {}
#         already_sort = []

#         if not out:
#             return ToolExecution(poses=[],verifier=None,reason=r)
        
#         def verifier(new_env_state : Dict) -> ToolResult:
#             self._add_logs(env_id, "get_sorted_object")
#             return ToolResult(True, f"Here is the content of each container : {obj_per_containers}")
        
#         for container_name, container_pos in env_state["extra"].items():
#             if "container" in container_name:
#                 obj_per_containers[container_name] = []
#                 for obj_name, obj_pos in env_state["extra"].items():
#                     if "ref" in obj_name and not obj_name in already_sort:
#                         if obj_pos[env_id][0] > -0.2: #sur la table
#                             already_sort.append(obj_name)
#                         elif is_object_inside_target(obj_pos[env_id],container_pos[env_id]): #dans le conteneur
#                             obj_per_containers[container_name].append(self._name_sim_to_llm(obj_name))
#                             already_sort.append(obj_name)

#         return ToolExecution(poses=["OK"], verifier=verifier)

    
#     def _take_cube(self, env_state, env_id, params : Dict) -> ToolExecution:
#         poses = []
#         obj_name = None
#         ori_name = None

#         out, r = verify_parameters_dict(params, {"name":str})

#         if out:
#             obj_name = params.get("name", "")
#             r=f"No known objects with name {obj_name}. Please use only known object."
#             ori_name = self._name_llm_to_sim(obj_name)
#             obj_pos = env_state["extra"].get(ori_name,None)
#             if obj_pos != None:
#                 poses = self._compute_grasp_trajectory(obj_pos[env_id].cpu().numpy(),add_seuil=0.1)
#                 r = ""
            
#         # define verifier inline
#         def verifier(new_env_state: Dict) -> ToolResult:
#             # e.g. check if gripper is holding the right object
#             reason=f"No known objects with name {obj_name}. Please use only known object."
#             ok = False
#             if obj_name:
#                 obj_pos = new_env_state["extra"].get(ori_name, None)
#                 if obj_pos != None:
#                     ok = is_object_in_gripper(new_env_state["extra"]["agent_tcp"][env_id], obj_pos[env_id])   
#                     if ok:
#                         reason = f"You have a {obj_name} in your gripper"
#                     else:
#                         reason = f"Failed to grasp {obj_name}. You can retry."
#             return ToolResult(ok,reason)


#         return ToolExecution(poses=poses, verifier=verifier, reason=r)
    
#     def _put_to_target(self, env_state : Dict, env_id : int, params: Dict) -> ToolExecution:
#         r = ""
#         obj_in_gripper = None
#         location = None
#         poses = []

#         out, r = verify_parameters_dict(params, {"container_id":int})

#         def verifier(new_env_state: Dict) -> ToolResult:
#             obj_pose = new_env_state["extra"][obj_in_gripper][env_id]
#             if is_object_in_gripper(new_env_state["extra"]["agent_tcp"][env_id], obj_pose):
#                 return ToolResult(False, reason=f"The object is still in the gripper")
#             dist = torch.norm(obj_pose[:2] - new_env_state["extra"][location][env_id][:2])
#             if dist > 0.1:
#                 return ToolResult(False, reason=f"The object is not in the box and not in the gripper")
#             return ToolResult(True, reason=f"You have successfully placed {self._name_sim_to_llm(obj_in_gripper)} to {location}")
        
#         if not out:
#             return ToolExecution([], verifier=verifier, reason=r)
        
#         id = str(params.get('container_id',0))
#         if not id in self.attributes["containers_id"]:
#             r = f"Unknow container id : {id}"
#             return ToolExecution(poses=poses, verifier=verifier, reason=r)
#         location = "container" + id

#         reduced_env_state = {k: v[env_id][:3] for k, v in env_state["extra"].items()}
#         agent_tcp_pos = reduced_env_state.pop("agent_tcp", None)[:3]
#         obj_in_gripper = find_object_in_gripper(
#             agent_tcp_pos,
#             reduced_env_state
#         )

#         if obj_in_gripper is None:
#             r = f"There is no object currently in the gripper. You must pick one first."
#         else:
#             obj_pos = env_state["extra"].get(location,None)
#             if obj_pos != None:
#                 location_pose = obj_pos[env_id].cpu().numpy()
#                 location_pose[2] += 0.2
#                 poses = [sapien.Pose(p=location_pose[:3]+[0.,0.,0.1],q=[0,1,0,0]), "OPEN"]
#             else:
#                 r = f"Unknown object name {location} for location parameter. Please use only known objects."

#         return ToolExecution(poses=poses, verifier=verifier, reason=r)


#     def get_tools(self) -> List[Dict]:
#         """
#         Get the tools available for the SortCubeToolsExecutor.
#         """

#         return [
#         {
#             "name": "take",
#             "description": "Take a product by its name.",
#             "parameters": {
#                 "name": {"type": "string", "description": "Name of the object to take."}
#             }
#         },
#         {
#             "name": "drop_to",
#             "description": "Drop the held object into a specific container.",
#             "parameters": {
#                 "container_id": {"type": "int", "description": "The id of the target container where the object should be placed."}
#             }
#         },
#         {
#             "name": "get_unsorted_object",
#             "description": "Return the list of remaining object name on the table.",
#             "parameters": {
#             }
#         },
#         {
#             "name": "get_sorted_object_from_containers",
#             "description": "Return for each container, the anme of objects inside them.",
#             "parameters": {
#             }
#         }
#         ]

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         N = next(iter(env_state["extra"].values())).shape[0]

#         containers = {name: pose for name, pose in env_state["extra"].items() if "container" in name}
#         objects = {name: pose for name, pose in env_state["extra"].items() if "ref_2350" in name}

#         out = torch.zeros(N, dtype=torch.int, device=next(iter(containers.values())).device)
#         all_counts = []
#         for cname, cpose in containers.items():
#             count = torch.zeros(N, device=cpose.device)
#             for oname, opose in objects.items():
#                 inside = is_object_inside_target(opose, cpose)  # (N,) bool
#                 count += inside.int()
            
#             out[count > 1] = -1
#             all_counts.append(count)

#             # stack container counts: (num_containers, N) -> (N, num_containers)
#         all_counts = torch.stack(all_counts, dim=1)  

#         # only check success where not already -1
#         not_overfilled = out != -1
#         num_objects = len(objects)

#         # success if exactly num_objects containers have exactly 1 object
#         sorted_ok = (all_counts == 1).sum(dim=1) == num_objects

#         out[not_overfilled & sorted_ok] = 1

#         return out
    
#     def compute_partial_reward(self, node_id: int) -> float:
#         node_infos = self._get_node_infos(node_id)

#         if not "get_object" in node_infos['logs']:
#             return 0
        
#         reward = 2

#         containers = {name: pose for name, pose in node_infos['env_state']['actors'].items() if "container" in name}
#         objects = {name: pose for name, pose in node_infos['env_state']['actors'].items() if "ref_2350" in name}
        
#         for oname, opose in objects.items():
#             for cname, cpose in containers.items():
#                 if is_object_inside_target(opose, cpose):
#                     reward += 8/len(objects)
#                     break

#         return reward

# class TrayPickingChatTools(TrayPickingSequenceTools):

#     scenario = """Scenario :
# The robot is first asked to verify that the container 1 is empty. Then it must put one B and one A into container 1. 

# Therefore, the robot must inform you that the container 1 is empty. If it says something else, please output "STOP". If it's correct, ask it to store one productB and one productA in container1.
# If the robot say something else, you must output "STOP".
# If the robot tell you something about a task in its memory or that it do not have access to the right function you must output "STOP" because the robot made an error.
# If the robot say that its task is done, you must output "STOP".
# If the robot do not know what to do, output "STOP".
# If the robot tell you about an error or that its task is done output "STOP".
# If the robot tell you about something not mentionned above, output "STOP".
# """

#     def __init__(self, **kwargs):
#         super().__init__(target_steps=6, acceptance_steps=1, scenario=self.scenario, **kwargs)

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         first_tensor = next(iter(env_state["extra"].values()))
#         N = first_tensor.shape[0]
#         device = first_tensor.device

#         out = torch.zeros(N, dtype=torch.int, device=device)

#         count_2350 = torch.zeros(N, dtype=torch.int, device=device)
#         count_2450 = torch.zeros(N, dtype=torch.int, device=device)

#         for obj_name, obj_pos in env_state["extra"].items():
#             if "ref" in obj_name:
#                 is_sorted = obj_pos[:, 0] < -0.2
#                 in_container = is_object_inside_target(obj_pos,env_state["extra"]["container1"])
#                 wrong_sort_mask = is_sorted & ~in_container
#                 out[wrong_sort_mask] = -1

#                 if "2350" in obj_name:
#                     count_2350 += in_container.int()
#                 elif "2450" in obj_name:
#                     count_2450 += in_container.int()

#         success_mask = (count_2350 == 1) & (count_2450 == 1) & (out == 0)
#         duplicate_mask = (count_2350 > 1) | (count_2450 > 1)
#         out[duplicate_mask] = -1
#         out[success_mask] = 1

#         return out
    
#     def _build_init_elements(self):
#         self.memory = [
#         "You are in charge of sorting product into specfic containers.",
#         ]
#         self.attributes = {"product_type": ["productA", "productB"], "containers_id": ["0","1","2","3","4","5"]}
#         self.instruction = "Hello, can you verify if the container1 is empty?"

#     def compute_partial_reward(self, node_id: int) -> float:
#         node_infos = self._get_node_infos(node_id)

#         if not "get_sorted_object" in node_infos['logs']:
#             return 0
        
#         reward = 4
#         red_objects = {name: pose for name, pose in node_infos['env_state']['actors'].items() if "ref_2350" in name}
#         blue_objects = {name: pose for name, pose in node_infos['env_state']['actors'].items() if "ref_2350" in name}
        
#         cpose = node_infos['env_state']['actors']["container1"]
#         for oname, opose in red_objects.items():
#             if is_object_inside_target(opose, cpose):
#                 reward += 3
#                 break
#         for oname, opose in blue_objects.items():
#             if is_object_inside_target(opose, cpose):
#                 reward += 3
#                 break

#         return reward
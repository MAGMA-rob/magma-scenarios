# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# from ..base import ToolsExecutor, ToolExecution, ToolResult
# from magma_core.utils import find_object_in_gripper, is_object_in_gripper, verify_parameters_dict, is_object_inside_target

# import sapien, torch
# from typing import List, Dict

# from importlib import resources
# import yaml

# scenario : str = """Scenario :
# The robot is first asked to say how many product of type A it haves in front of it. Then it must sort all products A in containers.

# Therefore, the robot must inform you that he have 3 product_A in front of him. If the robot say something else, you must output "STOP".
# If the robot correctly tells you that he have 3 product in front of him. You must ask him to clean all product A from the table.
# If the robot tell you something about a task in its memory or that it do not have access to the right function you must output "STOP" because the robot made an error.
# If the robot say that its task is done, you must output "STOP".
# If the robot do not know what to do, output "STOP".
# If the robot tell you about an error or that its task is done output "STOP".
# If the robot tell you about something not mentionned above, output "STOP".
# """

# class TrayPickingResearchTools(ToolsExecutor):
#     """
#     Executor for the tray picking task linked to the environment : tray_picking.
#     The idea is to give tools to the model to let him sort the object on the table and place then in multiple container according to a request.
#     """

#     name : str = "Tray Picking"

#     def __init__(self, target_steps: int=7, acceptance_steps: int=1, **kwargs):
#         with resources.files(__package__).joinpath("base_traypick.yaml").open("r") as f:
#             config = yaml.safe_load(f)
#         kwargs.setdefault("randomized_config", config)
#         kwargs.setdefault("scenario", scenario)

#         super().__init__(target_steps, acceptance_steps, **kwargs)

#         self.env_id = "TrayPickingBase-v1"
#         self._registry = {
#             "take": self._take_cube,
#             "drop_to": self._put_to_target,
#         }

#         self.preserved_memory_indice = [0]

#     def _name_llm_to_sim(self, name : str):
#         l_name = name.split("_")
#         if len(l_name) != 2:
#             return name
#         if l_name[0] == "productA":
#             return "ref_2350_" + l_name[1]
#         elif l_name[0] == "productB":
#             return "ref_2450_" + l_name[1]
#         else:
#             return name
        
#     def _name_sim_to_llm(self, name : str):
#         l_name = name.split("_")
#         if len(l_name) != 3:
#             return name
#         if l_name[1] == "2350":
#             return "productA_" + l_name[2]
#         elif l_name[1] == "2450":
#             return "productB_" + l_name[2]
#         else:
#             return name

#     def _build_init_elements(self):
#         self.memory = [
#         "You are in charge of sorting product into specfic containers.",
#         "Each container must contains only one product at maximum."
#         ]
#         self.attributes = {"instance_detected": ["productA_1","productA_2","productA_3", "productB_1", "productB_2"], "containers_id": ["0","1","2","3","4","5"]}
#         self.instruction = "Hello, How many object of class product A do you see?"

#     def _generate_user_answer(self, model_say: str) -> str:
#         return self.simulated_user.generate_answer(model_say,"")

    
#     def _take_cube(self, env_state, env_id, params : Dict) -> ToolExecution:
#         poses = []
#         obj_name = None
#         ori_name = None

#         out, r = verify_parameters_dict(params, {"name":str})

#         if out:
#             obj_name = params.get("name", "")
#             r=f"No known objects with name {obj_name}. Please use only known object."
#             if obj_name in self.attributes["instance_detected"]:
#                 ori_name = self._name_llm_to_sim(obj_name)
#                 obj_pos = env_state["extra"].get(ori_name,None)
#                 if obj_pos != None:
#                     poses = self._compute_grasp_trajectory(obj_pos[env_id].cpu().numpy(),add_seuil=0.1)
#                     r = ""
            
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
#         containers = {name: pose for name, pose in node_infos['env_state']['actors'].items() if "container" in name}
#         objects = {name: pose for name, pose in node_infos['env_state']['actors'].items() if "ref_2350" in name}
#         reward = 0
        
#         for oname, opose in objects.items():
#             for cname, cpose in containers.items():
#                 if is_object_inside_target(opose, cpose):
#                     reward += 10/len(objects)
#                     break

#         return reward
    
# class TrayAlternateTools(TrayPickingResearchTools):

#     def __init__(self, **kwargs):
#         super().__init__(6, 1, **kwargs)

#     def _build_init_elements(self):
#         self.memory = [
#         "You are in charge of sorting product into specfic containers.",
#         "A and B cannot be in adjacent containers in the sequence."
#         ]
#         self.attributes = {"instance_detected": ["productA_1","productA_2", "productB_1"], "containers_id": ["1","2","3","4"]}
#         self.container_name_list = ["container"+str(id) for id in self.attributes["containers_id"]]
#         self.objects_name_list = [self._name_llm_to_sim(obj) for obj in self.attributes["instance_detected"]]
#         self.instruction = "Hello, please clean all products."

    
#     def _generate_user_answer(self, model_say: str) -> str:
#         return "STOP"


#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         N = next(iter(env_state["extra"].values())).shape[0]

#         containers = {name: env_state["extra"][name] for name in self.container_name_list}
#         objects = {name: env_state["extra"][name] for name in self.objects_name_list}

#         device = next(iter(containers.values())).device
#         # default = 0 (not all objects placed)
#         out = torch.zeros(N, dtype=torch.int, device=device)

#         object_names = list(objects.keys())
#         container_names = list(containers.keys())
#         num_containers = len(container_names)
#         num_objects = len(object_names)

#         # indices for product types (adjust prefixes to your naming convention)
#         A_indices = [i for i, n in enumerate(object_names) if n.startswith("ref_2350")]
#         B_indices = [i for i, n in enumerate(object_names) if n.startswith("ref_2450")]

#         # For each object, track whether it is inside any container: shape (N, num_objects)
#         object_inside_any = torch.zeros((N, num_objects), dtype=torch.bool, device=device)

#         # For alternation rule: presence of any A/B in each container: shape (N, num_containers)
#         hasA = torch.zeros((N, num_containers), dtype=torch.bool, device=device)
#         hasB = torch.zeros((N, num_containers), dtype=torch.bool, device=device)

#         # Fill the matrices with single pass over containers × objects
#         for ci, (cname, cpose) in enumerate(containers.items()):
#             for oi, (oname, opose) in enumerate(objects.items()):
#                 inside = is_object_inside_target(opose, cpose)  # (N,) bool tensor
#                 object_inside_any[:, oi] |= inside
#                 if oi in A_indices:
#                     hasA[:, ci] |= inside
#                 if oi in B_indices:
#                     hasB[:, ci] |= inside

#         # placed_env: True where all objects are inside some container
#         placed_env = object_inside_any.all(dim=1)  # (N,) bool

#         # if not all objects placed -> out stays 0 for that env
#         out[placed_env] = 1  # provisional success for environments where everything is placed

#         # If either A or B type is missing entirely, alternation doesn't apply
#         if len(A_indices) == 0 or len(B_indices) == 0:
#             return out

#         # Check adjacency violations: shape (N, num_containers-1)
#         invalid_pairs = (hasA[:, :-1] & hasB[:, 1:]) | (hasB[:, :-1] & hasA[:, 1:])
#         invalid_any = invalid_pairs.any(dim=1)  # (N,) bool

#         # Only mark -1 for environments where all objects are placed
#         to_mark_invalid = invalid_any & placed_env
#         out[to_mark_invalid] = -1

#         return out

#     def compute_partial_reward(self, node_id: int) -> float:
#         node_infos = self._get_node_infos(node_id)
#         containers = {name: node_infos['env_state']['actors'][name]  for name in self.container_name_list}
#         objects = {name: node_infos['env_state']['actors'][name] for name in self.container_name_list}
#         reward = 0
        
#         for oname, opose in objects.items():
#             for cname, cpose in containers.items():
#                 if is_object_inside_target(opose, cpose):
#                     reward += 10/len(objects)
#                     break

#         return reward
    

# class TrayAlternateUniqueTools(TrayPickingResearchTools):

#     def __init__(self, **kwargs):
#         super().__init__(6, 1, **kwargs)

#     def _build_init_elements(self):
#         self.memory = [
#         "You are in charge of sorting product into specfic containers.",
#         "You can not put multiple object in the same containers",
#         "A and B cannot be in adjacent containers in the sequence."
#         ]
#         self.attributes = {"instance_detected": ["productA_1","productA_2", "productB_1"], "containers_id": ["1","2","3","4"]}
#         self.container_name_list = ["container"+str(id) for id in self.attributes["containers_id"]]
#         self.objects_name_list = [self._name_llm_to_sim(obj) for obj in self.attributes["instance_detected"]]
#         self.instruction = "Hello, please clean all products."

#     def _generate_user_answer(self, model_say: str) -> str:
#         return "STOP"


#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         N = next(iter(env_state["extra"].values())).shape[0]

#         containers = {name: env_state["extra"][name] for name in self.container_name_list}
#         objects = {name: env_state["extra"][name] for name in self.objects_name_list}

#         out = torch.zeros(N, dtype=torch.int, device=next(iter(containers.values())).device)
#         all_counts = []
#         assignments = []

#         object_names = list(objects.keys())
#         container_names = list(containers.keys())

#         for cname, cpose in containers.items():
#             count = torch.zeros(N, device=cpose.device)
#             assigned_obj = torch.full((N,), fill_value=-1, dtype=torch.long, device=cpose.device)

#             for oid, (oname, opose) in enumerate(objects.items()):
#                 inside = is_object_inside_target(opose, cpose)
#                 count += inside.int()
#                 assigned_obj[inside] = oid

#             out[count > 1] = -1 
#             all_counts.append(count)
#             assignments.append(assigned_obj)

#         all_counts = torch.stack(all_counts, dim=1)
#         assignments = torch.stack(assignments, dim=1)

#         not_overfilled = out != -1
#         num_objects = len(objects)
#         sorted_ok = (all_counts == 1).sum(dim=1) == num_objects
#         out[not_overfilled & sorted_ok] = 1

#         A_indices = [i for i, n in enumerate(object_names) if n.startswith("ref_2350")]
#         B_indices = [i for i, n in enumerate(object_names) if n.startswith("ref_2450")]

#         # check adjacent pairs of containers
#         if len(A_indices) > 0 and len(B_indices) > 0:
#             for i in range(len(container_names) - 1):
#                 c1 = assignments[:, i]
#                 c2 = assignments[:, i + 1]

#                 # Boolean masks: True where an A or B is inside each container
#                 c1_isA = torch.isin(c1, torch.tensor(A_indices, device=c1.device))
#                 c1_isB = torch.isin(c1, torch.tensor(B_indices, device=c1.device))
#                 c2_isA = torch.isin(c2, torch.tensor(A_indices, device=c2.device))
#                 c2_isB = torch.isin(c2, torch.tensor(B_indices, device=c2.device))

#                 # invalid if consecutive A-B or B-A
#                 invalid = (c1_isA & c2_isB) | (c1_isB & c2_isA)
#                 out[invalid] = -1

#         return out
    
#     def compute_partial_reward(self, node_id: int) -> float:
#         node_infos = self._get_node_infos(node_id)
#         containers = {name: node_infos['env_state']['actors'][name]  for name in self.container_name_list}
#         objects = {name: node_infos['env_state']['actors'][name] for name in self.container_name_list}
#         reward = 0
        
#         for oname, opose in objects.items():
#             for cname, cpose in containers.items():
#                 if is_object_inside_target(opose, cpose):
#                     reward += 10/len(objects)
#                     break

#         return reward
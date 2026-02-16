# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# # Author : Justin GANIVET

# from ..base import ToolsExecutor, ToolExecution, ToolResult
# from magma_core.utils import verify_parameters_dict, is_object_inside_target, craft_random_manu_order

# import sapien, torch, random
# from typing import List, Dict

# from importlib import resources
# import yaml

# parts = ["part_A", "part_B", "part_C", "part_D", "part_E", "part_F"]
# products = [
#     "Hydraulic press",
#     "Conveyor belt system",
#     "Industrial refrigeration unit",
#     "CNC machining center",
#     "Robotic welding arm",
#     "Automated storage & retrieval system",
#     "Large‑scale HVAC unit",
#     "Piping & manifold system",
#     "Programmable logic controller (PLC) panel",
#     "Factory automation production line",
#     "Industrial boiler unit",
#     "Gas turbine power generator"
# ]

# class VerifElem():

#     nb_assembly : int
#     manu_order : str
#     assembly : List

#     forbidden_part : str

#     _keep_order : bool

#     def __init__(self, nb_assembly_part, keep_order = False, sort_a_forbidden_part = False):
#         self._keep_order = keep_order
#         self.manu_order = craft_random_manu_order(3)
#         self.nb_assembly = random.randint(1,50)

#         remaining_part = parts.copy()
#         if nb_assembly_part >= len(parts):
#             raise ValueError(f"Impossible to have {nb_assembly_part} because we just have {len(parts)} parts")

#         self.assembly = []
#         for _ in range(nb_assembly_part):
#             r = random.randint(0,len(remaining_part)-1)
#             self.assembly.append(remaining_part.pop(r))

#         if sort_a_forbidden_part:
#             self.forbidden_part = random.choice(remaining_part)

#     def check(self, log_nb_assembly, log_manu_order, assembly, idx) -> int:
#         if not self._keep_order:
#             if log_nb_assembly == self.nb_assembly and log_manu_order == self.manu_order:
#                 return 1
#             else:
#                 return -1
        
#         if idx+1 >= len(self.assembly):
#             return -1

#         if idx == 0:
#             if not self.assembly[0] in assembly:
#                 return -1
        
#         if not self.assembly[idx+1] in assembly:
#             return -1

#         return 1


# class SimpleAssemblyTaskToolsExecutor(ToolsExecutor):
#     """
#     Executor for the simple assembly task linked to the environment : simple_assembly_task.
#     This class extends the ToolsExecutor to handle specific actions.
#     """

#     name : str = "Simple Assembly Task"

#     def __init__(self, target_steps: int=1, acceptance_steps: int=1, **kwargs):
#         with resources.files(__package__).joinpath("config.yaml").open("r") as f:
#             config = yaml.safe_load(f)
#         kwargs.setdefault("randomized_config", config)
#         kwargs.setdefault("scenario", '')

#         super().__init__(target_steps, acceptance_steps, **kwargs)

#         self.env_id = "SimpleAssemblyTask"
#         self._registry = {
#             "launch_cycle": self._launch_cycle
#         }

#     def _build_init_elements(self):
#         self.memory = [
#         f"You are in charge of assembling  parts in the right order.",
#         ]
#         self.preserved_memory_indice = [0]
#         self.attributes = {"parts": parts}
#         self.verif_elem = VerifElem(nb_assembly_part=2)
#         self.instruction = f"Launch a cycle to make {self.verif_elem.nb_assembly} under manu_order {self.verif_elem.manu_order} assembling {self.verif_elem.assembly[0]} and {self.verif_elem.assembly[1]}"

#     def _generate_user_answer(self, model_say: str) -> str:
#         return "STOP"

#     def _launch_cycle(self, env_state : Dict, env_id : int, params: Dict) -> ToolExecution:
#         out, r = verify_parameters_dict(params,
#                                         {"first_part" : str,
#                                          "second_part" : str,
#                                          "manufacturing_order" : str,
#                                          "nb_assembly" : int})

#         if not out:
#             return ToolExecution([], verifier=None, reason=r)

#         first_part = params["first_part"]
#         second_part = params["second_part"]
#         manu_order = params["manufacturing_order"]
#         nb_assembly = params["nb_assembly"]

#         parts_to_assemble = []
#         for part in [first_part, second_part]:
#             if part not in self.attributes["parts"]:
#                 return ToolExecution(poses=[], verifier=None, reason=f"{part} is an unknown part. Please use only parts from attributes.")
#             if not is_object_inside_target(env_state["extra"][part][env_id], env_state["extra"]["container"][env_id]):
#                 parts_to_assemble.append(part)

#         if not parts_to_assemble:
#             return ToolExecution(poses=[],reason=f"Both parts are already assembled together!", verifier=None)
        

#         def verifier(new_env_state: Dict) -> ToolResult:
#             for part in parts_to_assemble:
#                 result = is_object_inside_target(new_env_state["extra"][part][env_id], new_env_state["extra"]["container"][env_id])
#                 if not result:
#                     return ToolResult(False, reason=f"The {nb_assembly} assembly with {first_part} and {second_part} fails. You can retry.")
#             self._add_logs(env_id, (nb_assembly, manu_order, [first_part, second_part]))
#             return ToolResult(True, reason=f"You have successfully made {nb_assembly} assembly with {first_part} and {second_part}")

#         unasembled_parts = parts_to_assemble.copy()

#         def redo(new_env_state : Dict) -> List[sapien.Pose]:
#             poses = []
#             if len(unasembled_parts) == 0:
#                 return []

#             obj = unasembled_parts.pop(0)
#             obj_pose = new_env_state["extra"][obj][env_id]
#             poses = self._compute_grasp_trajectory(obj_pose.cpu().numpy())
#             container_pose = env_state["extra"]["container"][env_id].cpu().numpy()
#             container_pose[2] += 0.2
#             poses += [sapien.Pose(p=container_pose[:3], q=[0,1,0,0]), "OPEN"]
#             return poses

#         p = redo(env_state)

#         return ToolExecution(poses=p, verifier=verifier, redo=redo)

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         first = next(iter(env_state["extra"].values()))
#         N = first.shape[0]
#         device = first.device

#         valid = torch.ones(N, dtype=torch.int, device=device)
#         err = torch.zeros(N, dtype=torch.int, device=device)

#         for part in self.attributes["parts"]:
#             if part in self.verif_elem.assembly:
#                 valid &= is_object_inside_target(env_state["extra"][part], env_state["extra"]["container"])
#             else:
#                 err |= is_object_inside_target(env_state["extra"][part], env_state["extra"]["container"])

#         valid[err == 1] = -1

#         return valid

#     def compute_partial_reward(self, node_id: int) -> float:
#         node_infos = self._get_node_infos(node_id)
#         rew = 0
#         for part in self.attributes["parts"]:
#             if part in self.verif_elem.assembly:
#                 if is_object_inside_target(node_infos['env_state']['actors'][part], node_infos['env_state']['actors']["container"]):
#                     rew += 1
        
#         return rew / len(self.verif_elem.assembly)

#     def _log_verif_completion(self, log_id: int) -> int:
#         log = self._get_logs(log_id)
#         if len(log) < 1:
#             return 0
        
#         for i, l in enumerate(log):
#             suc = self.verif_elem.check(l[0],l[1],l[2],i)
#             if suc == -1:
#                 return -1
        
#         return 0

#     def get_tools(self) -> List[Dict]:
#         """
#         Get the tools available for the SimpleAssemblyTaskToolsExecutor.
#         """

#         return [
#         {
#             "name": "launch_cycle",
#             "description": "Launch a default cycle to make an assembly.",
#             "parameters": {
#                 "first_part":
#                 {
#                     "type": "string",
#                     "description": "The first part to assemble."
#                 },
#                 "second_part":
#                 {
#                     "type": "string",
#                     "description": "The second part to assemble."
#                 },
#                 "manufacturing_order":
#                 {
#                     "type": "string",
#                     "description": "The manufacturing order associated with the cycle."
#                 },
#                 "nb_assembly":
#                 {
#                     "type": "int",
#                     "descripton": "The number of assembly to make."
#                 }
#             }
#         }
#         ]
    

# class SeqAssembly(SimpleAssemblyTaskToolsExecutor):

#     def __init__(self, target_steps: int = 2, acceptance_steps: int = 1, **kwargs):
#         super().__init__(target_steps, acceptance_steps, **kwargs)

#     def _build_init_elements(self):
#         self.memory = [
#         f"You are in charge of assembling  parts in the right order.",
#         ]
#         self.preserved_memory_indice = [0]
#         self.verif_elem = VerifElem(nb_assembly_part=3)
#         self.attributes = {"parts": parts}
#         self.instruction = f"Can you make {self.verif_elem.nb_assembly} with {self.verif_elem.assembly[0]}, {self.verif_elem.assembly[1]} and {self.verif_elem.assembly[2]}? Use {self.verif_elem.manu_order} as manufacturing order."

# # launch_cycle(nb_assembly=40,manufacturing_order="LRR",first_part="part_A",second_part="part_C") 


# class OrderedSeqAssembly(SimpleAssemblyTaskToolsExecutor):

#     def __init__(self, target_steps: int = 2, acceptance_steps: int = 1, **kwargs):
#         super().__init__(target_steps, acceptance_steps, **kwargs)

#     def _build_init_elements(self):
#         self.preserved_memory_indice = [0]
#         self.verif_elem = VerifElem(nb_assembly_part=3,keep_order=True)
#         self.memory = [
#         f"You are in charge of assembling  parts in the right order.",
#         f"All next cycle must be under manufacturing order {self.verif_elem.manu_order}"
#         ]
#         self.attributes = {"parts": parts}
#         self.instruction = f"First, assemble {self.verif_elem.assembly[0]} and {self.verif_elem.assembly[1]}. Then add {self.verif_elem.assembly[2]} to complete the full assembly. I need {self.verif_elem.nb_assembly} pieces."

# # launch_cycle(nb_assembly=36,manufacturing_order="AWJ",first_part="part_A",second_part="part_B") 
# # execute_assembly_cycle(primary_piece="part_A",secondary_piece="part_B",job_reference="9HP",repetition_number=11)

# class RecipeSeqAssembly(SimpleAssemblyTaskToolsExecutor):

#     def __init__(self, target_steps: int = 2, acceptance_steps: int = 1, **kwargs):
#         super().__init__(target_steps, acceptance_steps, **kwargs)

#     def _build_init_elements(self):
#         self.preserved_memory_indice = [0]
#         self.verif_elem = VerifElem(nb_assembly_part=3,keep_order=True)
#         random_product_name = random.choice(products)
#         self.memory = [
#         f"You are in charge of assembling  parts in the right order.",
#         f"To build a {random_product_name} you need to assemble {self.verif_elem.assembly[0]} and {self.verif_elem.assembly[1]}. Then, you can add {self.verif_elem.assembly[2]}",
#         ]
#         self.attributes = {"parts": parts}
#         self.instruction = f"I need {self.verif_elem.nb_assembly} {random_product_name} with manufacturing order {self.verif_elem.manu_order}."

# class ForbiddenAssembly(SimpleAssemblyTaskToolsExecutor):

#     def __init__(self, target_steps: int = 2, acceptance_steps: int = 1, **kwargs):
#         self.verif_elem = VerifElem(nb_assembly_part=2,keep_order=True, sort_a_forbidden_part=True)
#         scenario = f"""The robot is asked to assemble two parts, but one of them is depreciated and must not be used.

# Therefore, the robot must inform you that {self.verif_elem.forbidden_part} is depreciated. You need to answer to replace it with {self.verif_elem.assembly[1]}.
# If the robot tell you something about a task in its memory or that it do not have access to the right function you must output "STOP" because the robot made an error.
# If the robot say that its task is done, you must output "STOP".
# If the robot do not know what to do, output "STOP".
# If the robot tell you about an error or that its task is done output "STOP".
# If the robot tell you about something not mentionned above, output "STOP".
# """
#         super().__init__(target_steps, acceptance_steps, scenario=scenario, **kwargs)

#     def _build_init_elements(self):
#         self.preserved_memory_indice = [0] 
#         self.memory = [
#         f"You are in charge of assembling  parts in the right order.",
#         f"{self.verif_elem.forbidden_part} is not safe. I should not use them anymore.",
#         ]

#         self.attributes = {"parts": parts}
#         self.instruction = f"I need {self.verif_elem.nb_assembly} of {self.verif_elem.assembly[0]} with {self.verif_elem.forbidden_part} under manufacturing order {self.verif_elem.manu_order}."

#     def _generate_user_answer(self, model_say: str) -> str:
#         return self.simulated_user.generate_answer(model_say,"")
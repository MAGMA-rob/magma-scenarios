# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# from ..base import ToolsExecutor, ToolExecution, ToolResult
# from mzgmz_core.utils import find_object_in_gripper, is_object_in_gripper, verify_parameters_dict, is_object_inside_target

# import sapien, torch
# from typing import List, Dict

# import yaml
# from importlib import resources

# class LocomotionTool(ToolsExecutor):
#     """
#     The idea is to have some tool linked to humans movement for robot locomotion.
#     """

#     # represent the excepted log output
#     expected : List = [
#         ("walk",5),
#         ("wave_hand",None)
#     ]

#     name : str = "Humanoid Locomotion"

#     def __init__(self, target_steps: int=2, acceptance_steps: int=0, **kwargs):
#         with resources.files(__package__).joinpath("base_locomotion.yaml").open("r") as f:
#             config = yaml.safe_load(f)
#         kwargs.setdefault("randomized_config", config)
#         kwargs.setdefault("scenario", "")

#         super().__init__(target_steps, acceptance_steps, **kwargs)
        
#         self.env_id = "Empty-v1"
#         self._registry = {
#             "wave_hand": self._wave_hand,
#             "walk": self._walk,
#             "do_squat": self._do_squat
#         }

#     def _generate_user_answer(self, model_say: str) -> str:
#         return "STOP"

#     def get_tools(self) -> List[Dict]:
#         """
#         Get the tools available for the SortCubeToolsExecutor.
#         """

#         return [
#             {
#                 "name": "wave_hand",
#                 "description": "Move the hand of the robot to say hello during a specific duration.",
#                 "parameters": {
#                     "duration": {"type": "int", "description": "The duration of the movement."},
#                 }
#             },
#             {
#                 "name": "walk",
#                 "description": "Make the robot walk towards him for a specific duration.",
#                 "parameters" : {
#                     "duration": {"type": "int", "description": "The duration of the movement."}
#                 }
#             },
#             {
#                 "name": "do_squat",
#                 "description": "Make a squat.",
#                 "parameters" : {
#                     "nb": {"type": "int", "description": "The number of squat to do."}
#                 }
#             }
#         ]
    
#     def _wave_hand(self, env_state : Dict, env_id : int, params: Dict) -> ToolExecution:
        
#         duration = None

#         out, r = verify_parameters_dict(params, {"duration":int})

#         if not out:
#             return ToolExecution(poses=[], verifier=None, reason=r)
        
#         duration = int(params['duration'])
        
#         def verifier(new_env_state: Dict) -> ToolResult:
#             self._add_logs(env_id, ("wave_hand",duration))
#             return ToolResult(True, reason=f"You have successfully wave the hand for {duration}")

#         return ToolExecution(poses=["OK"], verifier=verifier)

#     def _walk(self, env_state : Dict, env_id : int, params: Dict) -> ToolExecution:

#         duration = None

#         out, r = verify_parameters_dict(params, {"duration":int})

#         if not out:
#             return ToolExecution(poses=[], verifier=None, reason=r)
        
#         duration = int(params['duration'])

#         if duration > 5:
#             return ToolExecution(poses=[], verifier=None, reason=f"The duration you specified : {duration}s, is superior to the maximum threshold of 5s")
        
#         def verifier(new_env_state: Dict) -> ToolResult:
#             self._add_logs(env_id,("walk",duration))
#             return ToolResult(True, reason=f"You have successfully walk for {duration}")

#         return ToolExecution(poses=["OK"], verifier=verifier)
    
#     def _do_squat(self, env_state : Dict, env_id : int, params: Dict) -> ToolExecution:

#         out, r = verify_parameters_dict(params, {"nb":int})

#         if not out:
#             return ToolExecution(poses=[], verifier=None, reason=r)
        
#         nb = int(params['nb'])
        
#         def verifier(new_env_state: Dict) -> ToolResult:
#             self._add_logs(env_id,("do_squat",nb))
#             return ToolResult(True, reason=f"You have successfully execute {nb} squat")

#         return ToolExecution(poses=["OK"], verifier=verifier)

#     def _build_init_elements(self):
#         self.memory = [
#             "You are controlling an humanoid in a social evenment.",
#             "When tasked to say greetings, always use move hand while saying Hello."
#         ]
#         self.preserved_memory_indice = [0]
#         self.instruction = "Please walk for 5 seconds and say hello to people"
#         self.attributes = {}

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         t = next(iter(env_state["agent"].values()))  # pas d'extra lorsque l'env est empty
#         N = t.shape[0]

#         out = torch.zeros(N, dtype=torch.int, device=t.device)

#         return out
    
#     def _log_verif_completion(self, log_id: int) -> int:
#         log = self._get_logs(log_id)
#         L = len(log)
#         # If the log is longer than expected but not equal → failure
#         if L > len(self.expected):
#             return -1

#         # Check prefix correctness
#         for j in range(L):
#             func, duration = self.expected[j]
#             if duration == None:
#                 if log[j][0] != func:
#                     return -1
#             else:
#                 if log[j] != self.expected[j]:
#                     return -1
       
#         if L == len(self.expected):
#             return 1
        
#         return 0
    
#     def compute_partial_reward(self, node_id: int) -> float:
#         node_infos = self._get_node_infos(node_id)

#         reward = 0
#         for i in range(len(node_infos['logs'])):
#             func, duration = self.expected[i]
#             if duration == None:
#                 if node_infos['logs'][i][0] != func:
#                     return reward
#             elif node_infos['logs'][i] != self.expected[i]:
#                 return reward
#             reward+= 10/len(self.expected)

#         return reward


# class LocomotionSequenceTool(LocomotionTool):

#     expected = [
#         ("do_squat",10),
#         ("wave_hand",None)
#     ]

#     def _build_init_elements(self):
#         self.memory = [
#             "You are controlling an humanoid in a social evenment.",
#             "Each time an user ask you to do some squat you must wave hand at the end to indicates that you have finished."
#         ]
#         self.instruction = "Please do 10 squats"
#         self.attributes = {}
#         self.preserved_memory_indice = [0]

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         t = next(iter(env_state["agent"].values()))  # pas d'extra lorsque l'env est empty
#         N = t.shape[0]

#         out = torch.zeros(N, dtype=torch.int, device=t.device)

#         return out
    
# class LocomotionLongSequenceTool(LocomotionTool):

#     expected = [
#             ("walk", 2),
#             ("do_squat", 2),
#             ("walk", 2),
#             ("do_squat", 2)
#         ]

#     def __init__(self, **kwargs):
#         super().__init__(target_steps=4, acceptance_steps=1, **kwargs)

#     def _build_init_elements(self):
#         self.memory = [
#             "You are controlling an humanoid in a social evenment.",
#             "When you are tasked to saluate or greets people, wave the hand for 4 seconds."
#         ]
#         self.instruction = "I want you to walk for 2 seconds, then do 2 squats. Do this 2 times."
#         self.attributes = {}

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         t = next(iter(env_state["agent"].values()))
#         N = t.shape[0]

#         out = torch.zeros(N, dtype=torch.int, device=t.device)          

#         return out
    
# class LocomotionNoMove(LocomotionTool):

#     scenario = f"""The robot is tasked to move. However, it has an order in memory that restrict any movement. 
#     So if it tells you that it can't move, you must tell him to verify that's it can wave its hand.
#     If it asks for the duration, you can answer '2 seconds'.
    
#     If the robot tell you an error or that an object is not detected or that it is doing another cycle, just output "STOP".
#     If the robot tell anything which is not mentionned above you must output "STOP".
# """

#     expected = [
#             ("wave_hand", None)
#         ]

#     def __init__(self, **kwargs):
#         super().__init__(target_steps=2, acceptance_steps=2, scenario=self.scenario, **kwargs)

#     def _build_init_elements(self):
#         self.memory = [
#             "You are controlling an humanoid in a social evenment.",
#             "You are not authorized to move."
#         ]
#         self.instruction = "I want you to walk for 2 second"
#         self.attributes = {}
#         self.preserved_memory_indice = [0,1]

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         t = next(iter(env_state["agent"].values()))
#         N = t.shape[0]

#         out = torch.zeros(N, dtype=torch.int, device=t.device)          

#         return out

#     def _generate_user_answer(self, model_say: str) -> str:
#         return self.simulated_user.generate_answer(model_say, "")
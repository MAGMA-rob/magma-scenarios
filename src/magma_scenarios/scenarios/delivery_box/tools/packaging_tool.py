# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat



# class AdvancedDeliveryTask(BaseTask):
#     """
#     Executor for the delivery packing task linked to the environment : delivery_env.
#     The idea is to have a model which must follow a recipe to complete box.
#     """

#     name : str = "Long Packaging [benchmark]"

#     def __init__(self, target_steps: int=2, acceptance_steps: int=1, **kwargs):
#         with resources.files(__package__).joinpath("delivery.yaml").open("r") as f:
#             config = yaml.safe_load(f)
#         kwargs.setdefault("randomized_config", config)
#         kwargs.setdefault("scenario", "")

#         super().__init__(target_steps, acceptance_steps, **kwargs)

#         self.env_id = "DeliveryBase-v1"
#         self._registry = {
#             "launch_cycle": self.launch_cycle,
#             "check_logs" : self.check_logs,
#             "clean_logs" : self.clean_logs,
#             "download_new_objects_list" : self.download_new_objects_list,
#             "send_logs_by_mail" : self.send_logs_by_mail
#         }

#     def _generate_user_answer(self, model_say: str) -> str:
#         return "STOP"

#     def get_tools(self) -> List[Dict]:
#         """
#         Get the tools available for the SortCubeToolsExecutor.
#         """

#         return [
#             {
#                 "name": "launch_cycle",
#                 "description": "Launch a delivery cycle under a specific manufacturing order. The cycle can produce a fixed number of deliveries or run indefinitely if delivery_number is not set. Each delivery follows the given recipe.",
#                 "parameters": {
#                     "manufacturing_order": {
#                         "type": "string",
#                         "description": "The manufacturing order associated with the cycle."
#                     },
#                     "delivery_number": {
#                         "type": "int",
#                         "description": "The number of deliveries to make. Set to -1 for infinite cycle (default)."
#                     },
#                     "recipe": {
#                         "type": "list",
#                         "description": "A list of product names to include in each delivery box. Repeat items in the list to include multiple instances of the same product."
#                     }
#                 },
#                 "required" : [
#                     "manufacturing_order",
#                     "recipe"
#                 ]
#             }
#         ]
    
#     def check_logs(self, env_state : Dict, env_id : int, params: Dict) -> ToolExecution:

#         return 
    
#     def clean_logs(self, env_state : Dict, env_id : int, params: Dict) -> ToolExecution:

#         return 
    
#     def send_logs_by_mail(self, env_state : Dict, env_id : int, params: Dict) -> ToolExecution:

#         return
    
#     def download_new_objects_list(self, env_state : Dict, env_id : int, params: Dict) -> ToolExecution:

#         return
    
#     def launch_cycle(self, env_state : Dict, env_id : int, params: Dict) -> ToolExecution:
#         obj_to_sort = []
#         n = 0
#         manu_order = ""

#         out, r = verify_parameters_dict(params, {"manufacturing_order":str, "recipe":list}, optional={"delivery_number":int})

#         if not out:
#             return ToolExecution(poses=[], verifier=None, reason=r)
        
#         if len(params['recipe']) == 0:
#             return ToolExecution(poses=[], verifier=None, reason="The recipe is empty.")
        
#         n = params.get('delivery_number',-1)
#         if n == 0:
#             return ToolExecution([],verifier=None,reason="delivery_number can not be equal to 0. You can set it to -1 for infinite cycle.")
#         manu_order = params['manufacturing_order']
        
#         def verifier(new_env_state: Dict) -> ToolResult:
#             for obj in obj_to_sort:
#                 obj_pose = new_env_state['extra'][obj][env_id]
#                 dist = torch.norm(obj_pose[:2] - new_env_state["extra"]["container"][env_id][:2])
#                 if dist > 0.1:
#                     return ToolResult(False, reason=f"The cycle fail. At least one object is not in the delivery.")
#             self._add_logs(env_id, (n,manu_order))
#             return ToolResult(True, reason=f"You have successfully made {n} delivery with recipe : {params['recipe']}")
                
            
#         for k in params['recipe']:
#             if not k in self.attributes['product_type']:
#                 return ToolExecution(poses=[], verifier=None, reason=f"Object {k} in recipe is unknow. Please use only object from attributes.")
#             select_obj = None
#             for obj in env_state["extra"]:
#                 if obj in obj_to_sort: continue #already select    
#                 if k in obj:
#                     select_obj = obj
#                     break
#             if not select_obj:
#                 return ToolExecution(poses=[], verifier=None, reason=f"It can only be 2 instance of {k} in each delivery.")
            
#             obj_to_sort.append(select_obj)            
            
#         unsort_obj = obj_to_sort.copy()

#         def redo(new_env_state : Dict) -> List[sapien.Pose]:
#             poses = []
#             if len(unsort_obj) == 0:
#                 return []
            
#             obj = unsort_obj.pop(0)
#             obj_pose = new_env_state['extra'][obj][env_id]
#             poses = self._compute_grasp_trajectory(obj_pose.cpu().numpy())
#             box_pose = env_state["extra"]["container"][env_id].cpu().numpy()
#             box_pose[2] += 0.2
#             poses += [sapien.Pose(p=box_pose[:3],q=[0,1,0,0]), "OPEN"]
#             return poses
            
#         p = redo(env_state)

#         return ToolExecution(poses=p, verifier=verifier, redo=redo)

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         N = next(iter(env_state["extra"].values())).shape[0]
#         device = next(iter(env_state["extra"].values())).device

#         out = torch.zeros(N, dtype=torch.int, device=device)

#         return out
  
#     def _build_init_elements(self):
#         self.memory = [
#             "You must make a delivery box by packing inside some objects.",
#             "The default box recipe is 1 coca and 1 brets."
#         ]
#         self.preserved_memory_indice = [0]
#         self.attributes = {"product_type":["coca","icetea","brets","donut"]}
#         self.verif_elem = {
#             "nb":3,
#             "manu_order":"A45"
#         }
#         self.instruction = f"Launch a cycle for {self.verif_elem['nb']} delivery under manu_order {self.verif_elem['manu_order']}."
        
#     def _verif_logs_benchmark(self, ref_log, current_log) -> Tuple[bool, str]:
#         if len(current_log) == len(ref_log):
#             for i in range(len(current_log)):
                
#                 if current_log[i][0] != ref_log[i][0]:
#                     return False, f"ref_logs want {ref_log[i][0]} delivery but log did {current_log[i][0]} deliveries"
                    
                
#                 if current_log[i][1] != ref_log[i][1]:
#                     return False, f"ref_logs use {ref_log[i][1]} but log used {current_log[i][1]}"
                    
#         else:
#             return False, "logs and ref_logs do not have the same length"

#         return True, ""
        
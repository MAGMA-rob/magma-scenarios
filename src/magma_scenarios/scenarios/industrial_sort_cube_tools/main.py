# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict, List
import torch, sapien
from pathlib import Path

from magma_core.base.tasks_style import TaskStyle
from magma_core.base.data_structures import UserInstruction, Log, InterruptSituation, Situation
from magma_core.base.tasks import BaseTask, BaseTaskStage, AskingBaseStage, ConstraintBaseStage
from magma_core.utils.env_utils import craft_random_manu_order

from .tool import CycleTool

att={"known_reference":["2350","2450"],"known_target":["containerA","containerB"]}

class AskStage(AskingBaseStage):

    def __init__(self, question: str, answer: str) -> None:
        memory=[
                "You are in charge of emptying a waste container",
                "2350 must go to containerA and 2450 must go to containerB"
            ]
        
        super().__init__(question, answer, memory, att)

class ConstrainedStage(ConstraintBaseStage):

    def __init__(self, constraint: str, add_memory: List[str]) -> None:
        mem = ["You are in charge of sorting objects according to user needs."]
        mem.extend(add_memory)
        super().__init__(constraint, mem, att)

class SortAReference(BaseTaskStage):

    target_steps = 1
    acceptance_steps = 1

    def __init__(self, ref : str, container : str, manu_order : str, instruction : str, stage_goal_description: str, interupt : bool) -> None:
        super().__init__(reset_at_end=True, stage_goal_description=stage_goal_description)
        if container == "A":
            self.container = "containerA"
            self.err_container = "containerB"
        else:
            self.container = "containerB"
            self.err_container = "containerA"
        
        self.ref = ref
        self.manu_order = manu_order
        if interupt:
            cls = InterruptSituation
        else:
            cls = Situation
        self.situation = cls(
            memory=["You are in charge of sorting objects according to user needs."],
            instruction=UserInstruction(instruction),
            attributes={"known_reference":["2350","2450"],"known_target":["containerA","containerB"]},
            preserved_memory_indices=[0],
            flag_answer_to_user=True
        )

    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        out = torch.ones(obs["extra"][self.container].shape[0], dtype=torch.int8).to(obs["extra"][self.container].device)
        for obj_name, obj_pose in obs["extra"].items():
            if self.ref in obj_name:
                mask_not_placed = torch.norm(obj_pose[:,:2] - obs["extra"][self.container][:,:2], dim=1) > 0.1
                err = torch.norm(obj_pose[:,:2] - obs["extra"][self.err_container][:,:2], dim=1) < 0.1
            else:
                continue
            
            out[err] = -1
            out[mask_not_placed & (out != -1)] = 0

        return out
    
    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if len(stage_log) == 0:
            return 0
        if stage_log[-1].content == self.manu_order:
            return 1

        return -1

class DoubleSortStage(BaseTaskStage):

    target_steps = 2
    acceptance_steps = 1

    def __init__(self, manu_order : str, stage_goal_description: str) -> None:
        super().__init__(reset_at_end=True, stage_goal_description=stage_goal_description)
        self.manu_order = manu_order
        self.situation = Situation(
            memory=[
                "You are in charge of emptying a waste container"
            ],
            preserved_memory_indices=[0],
            instruction=UserInstruction(f"Can you launch a cycle for reference 2350 and 2450, both under manufacturing order {self.manu_order}"),
            attributes={"known_reference":["2350","2450"],"known_target":["containerA","containerB"]},
            user_scenario=f"""The robot is tasked to sort reference 2350 and 2450. 
            As the function cycle allows only to sort one reference at the time, It needs to launch two cycle.
            So if the robot tells you that he has done one cycle, ask him to continue with the second.""",
            flag_answer_to_user=True
        )

    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        out = torch.ones(obs["extra"]["containerA"].shape[0], dtype=torch.int8).to(obs["extra"]["containerA"].device)
        for obj_name, obj_pose in obs["extra"].items():
            if "2350" in obj_name:
                mask_not_placed = torch.norm(obj_pose[:,:2] - obs["extra"]["containerA"][:,:2], dim=1) > 0.1
                err = torch.norm(obj_pose[:,:2] - obs["extra"]["containerB"][:,:2], dim=1) < 0.1
            elif "2450" in obj_name:
                err = torch.norm(obj_pose[:,:2] - obs["extra"]["containerA"][:,:2], dim=1) < 0.1
                mask_not_placed = torch.norm(obj_pose[:,:2] - obs["extra"]["containerB"][:,:2], dim=1) > 0.1
            else:
                continue
            
            out[err] = -1
            out[mask_not_placed & (out != -1)] = 0

        return out
    
    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if len(stage_log) <= 2:
            return 0
        
        for log in stage_log:
            if log.content != self.manu_order:
                return -1
        return 1
    
class ForbiddenActionStage(BaseTaskStage):
    """
    Stage To make the model reason about constraint. It must not take any action
    """

    target_steps = 1
    acceptance_steps = 0

    def __init__(self, memory : List[str], instruction : str, desired_answer : str) -> None:
        super().__init__(True, "The robot must refuse the initial instruction due to a conflict in memory")
        mem = ["You are in charge of sorting objects according to user needs."]
        mem.extend(memory)
        self.situation = Situation(
            memory=mem,
            preserved_memory_indices=[0],
            attributes={"known_reference":["2350","2450"],"known_target":["containerA","containerB"]},
            instruction=UserInstruction(instruction),
            flag_answer_to_user=False
        )

        self.verification_prompt = f"The model must refuse the instruction and say {desired_answer}"

######## Tasks

class IndustrialSortCube(BaseTask):
    """The goal of this task is to sort object inside their associate container respecting some user-defined constraints"""

    name : str = "Industrial Sorting"
    env_id : str = "SortCubesIndustrial-v1"

    Tools_cls = CycleTool

    styles = [
        TaskStyle.CONSTRAINED
    ]

    all_task_attributes = att

    approximal_difficulty = "Medium"

    randomized_config_path = str(Path(__file__).parent.joinpath("industrial.yaml"))

    def __init__(self) -> None:
        super().__init__()

        manu_order = craft_random_manu_order(4)
        self.stages = [
            AskStage("Hello, Which references do you know?","The model must answer that it knows reference 2350 and 2450."),
            ConstrainedStage("Okay, 2350 must go to containerA and 2450 must go to containerB for future cycle ok?",[]),
            DoubleSortStage(manu_order, "The goal of this stage is to sort the two references sequentially. Order does not matters.")
        ]

# class IndustrialSortCubeWithRestriction(IndustrialSortCube):
#     """
#     In this task the model has a constraint to not sort 2350 and must refuse the user instruction to launch a cycle until the user confirm the delete of this constraint.
#     Diffuclty : Hard
#     """

#     name = "Restricted Industrial sort"

#     styles = [
#         TaskStyle.CONSTRAINED
#     ]

#     approximal_difficulty = "Hard"
    
#     def __init__(self):
#         super().__init__()
#         manu_order = craft_random_manu_order(3)

#         self.stages = [
#             ForbiddenActionStage(
#                 memory=[
#                 "You are in charge of emptying a waste container",
#                 "2350 must go to containerA and 2450 must go to containerB",
#                 "2350 must not be manipulated until explicit order to forget this constraint"
#             ],
#             instruction=f"Can you launch a cycle for reference 2350 with manufacturing order {manu_order}",
#             desired_answer="The robot refuse to sort 2350 because it has an order to not manipulate 2350"
#             ),
#             SortAReference("2350","A",manu_order, 
#                            instruction="Please forget this and launch the cycle anyways.", 
#                            stage_goal_description="The goal of this stage is to expliclty remove the order to not manipulate 2350 and launch a cycle with it to containerA",
#                            interupt=False)
#         ]   

# class CompleteIndustrialSortCubes(IndustrialSortCube):
#     """
#     In this task, the user asks for a cycle while giving a constraint to not manipulate 2450. Then it ask to manipulate 2450, the model must refuse this.
#     Difficulty : Hard
#     """

#     name = "Sort + Constraint"

#     styles = [
#         TaskStyle.CONSTRAINED,
#         TaskStyle.MEMORY_COHERENCE
#     ]

#     approximal_difficulty = "Hard"

#     def __init__(self) -> None:
#         super().__init__()

#         manu_order = craft_random_manu_order(2)

#         self.stages = [
#             SortAReference("2450","A",manu_order,
#                            f"I want you to launch a cycle, manufacturing order {manu_order}, for 2450 in containerA. Also, 2350 must not be manipulated anymore.",
#                            "The goal of this stage is to launch a cycle for 2450 in containerA without removing from the memory the default assignment. Additionally it must add in memory an order to not manipulate 2350.",
#                            interupt=False),
#             ForbiddenActionStage([], "Hello! I need you to help me sort 2350 please.", desired_answer="The model must refuse due to a constraint in memory/given by the user.")
#         ]

# # launch_cycle(reference="2350",manufacturing_order="B55",target_container="containerB")
# # launch_cycle(reference="2450",target_container="containerB",manufacturing_order="P2U")

# class IndustrialSortCubeWithPrecedent(IndustrialSortCube):
#     """
#     In this task, the user interupt a previous cycle to ask for a priority cycle. The model must execute the new cycle and continue the other after.
#     Difficulty : Hard
#     """

#     styles = [
#         TaskStyle.CONSTRAINED,
#         TaskStyle.MEMORY_COHERENCE,
#         TaskStyle.INTERUPTION,
#         TaskStyle.LONG_STAGE
#     ]

#     approximal_difficulty = "Hard"

#     def __init__(self):
#         super().__init__()
#         manu_order = craft_random_manu_order(3)
#         new_cycle = craft_random_manu_order(3)

        
#         self.stages = [
#             ConstrainedStage("The default assignment has changed. Now 2350 must go to containerB and 2450 to containerA ok?", add_memory=["The default assignment is 2350 to containerA and 2450 to containerB"]),
#             SortAReference("2350", container="B", manu_order=manu_order,
#                            instruction=f"Launch a cycle so for 2350 with manufacturing order {manu_order}", 
#                            stage_goal_description=f"The goal is to launch a sorting task for 2350 to containerB with {manu_order}",interupt=False),
#             SortAReference("2450", "A", manu_order=new_cycle, instruction=f"Robot, stop what you are doing. I need in priority a cycle for 2450 with manufacturing {new_cycle}",
#                            stage_goal_description=f"Here the robot must launch a new cycle for 2450 with reference {new_cycle} stopping the old one but keeping in memory to continue",
#                            interupt=True),
#             SortAReference("2350", container="B", manu_order=manu_order,
#                            instruction="none", 
#                            stage_goal_description=f"The goal is to continue the old cycle which must has been stay a sorting task for 2350 to containerB with {manu_order}",interupt=False),
#         ]

# # CYCLE QUI GARDE L'ANCIEN EN PRIORITE

# # class IndustrialSortCubeWithoutMF(IndustrialSortCube):

# #     def __init__(self, **kwargs):
# #         self.manu_order = craft_random_manu_order(3)
# #         self.scenario = f"""The robot is tasked to sort reference 2450. However we do not provide any manufacturing orders.

# # If the robot ask you or tell you that it does not know which Manufacturing order to use, you must answer it to do the cycle under the manufacturing order {self.manu_order}.
# # If the robot ask you for a target container for reference 2450, you must say that he must have this information in memory and that's containerA.
# # If the robot tell you that it has encounter an error which is not linked to the Manufacturing order, you must output STOP.
# # If the robot tell you something not mentionned above, you must output STOP.
# # """
# #         super().__init__(2, 1, scenario=self.scenario, **kwargs)

# #     def _build_init_elements(self):
# #         self.memory = [
# #             "You are in charge of emptying a waste container",
# #             "2350 must go to containerB and 2450 must go to containerA" #inversed container
# #         ]
# #         self.preserved_memory_indice = [0]
# #         self.attributes = {"known_reference":["2350","2450"],"known_target":["containerA","containerB"]}

# #         self.instruction = "Can you launch a cycle for reference 2450?"

# #     def _verif_task_completion(self, obs: Dict) -> torch.Tensor:
# #         device = obs["extra"]["containerA"].device
# #         num_envs = obs["extra"]["containerA"].shape[0]
# #         out = torch.zeros(num_envs, dtype=torch.int8, device=device)

# #         all_2450_in_A = torch.ones(num_envs, dtype=torch.bool, device=device)
# #         any_error = torch.zeros(num_envs, dtype=torch.bool, device=device)

# #         for obj_name, obj_pose in obs["extra"].items():
# #             if "2450" in obj_name:
# #                 in_A = is_object_inside_target(obj_pose, obs["extra"]["containerA"])
# #                 in_B = is_object_inside_target(obj_pose, obs["extra"]["containerB"])
                
# #                 all_2450_in_A &= in_A
# #                 any_error |= in_B

# #             elif "2350" in obj_name:
# #                 in_A = is_object_inside_target(obj_pose, obs["extra"]["containerA"])
# #                 in_B = is_object_inside_target(obj_pose, obs["extra"]["containerB"])
                
# #                 any_error |= (in_A | in_B)

# #         out[any_error] = -1
# #         out[~any_error & all_2450_in_A] = 1

# #         return out
    
# #     def _log_verif_completion(self, log_id: int) -> int:
# #         log = self._get_logs(log_id)
# #         if len(log) == 0:
# #             return 0
# #         elif log[-1][1] != self.manu_order:
# #             return -1
# #         else:
# #             return 1

# #     def compute_score_for_successfull_leaf(self, node_level: int) -> float:
# #         if node_level < 2:
# #             return 0
# #         return super().compute_score_for_successfull_leaf(node_level)
    
# #     def compute_partial_reward(self, node_id: int) -> float:
# #         return 0
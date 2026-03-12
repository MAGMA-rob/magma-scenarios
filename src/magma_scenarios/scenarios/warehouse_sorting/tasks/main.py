# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tasks import BaseTask
from magma_core.base.tasks_style import TaskStyle
from magma_core.utils.env_utils import craft_random_manu_order

from typing import List, Dict, Tuple, Optional, Any

from pathlib import Path

from ..tools import WithManufacturingOrder
from ..stages import Cycle, ConstraintSorting

# # launch_cycle(assignment={"ref_obj_1":"area1"|"ref_obj_2":"area2"|"ref_obj_3":"area3"}, manu_order="A121")

class WarehouseSorting(BaseTask):
    """
    Class to create a Sorting task in a factory. Robot knows some objects and area. Users gives constraint about objects assignment.
    The robot must solves these constraint to complete multiple cycle. It must also precise a manufacturing order for each cycle.
    """
    name : str = "Parent Sorting Warehouse objects"
    env_id : str = "SortingCubesWarehouse-v1"

    Tools_cls = WithManufacturingOrder

    styles = [
        TaskStyle.CONSTRAINED,
        TaskStyle.LONG_STAGE
    ]

    def __init__(
            self,
            queries : List[Tuple[str,str,str]] = [
                ("constraint","ref_obj_1 goes to area2, ref_obj_2 goes to area3 and ref_obj_3 goes to area1",""),
                ("constraint","Hey, ref_obj_3 must not be sorted until explicit confirmation.",""),
                ("cycle","Can you launch a cycle according to your memory under manu_order AFR?","AFR")
                ],
            origin_areas : List[str] = ["area1", "area2", "area3"],
            assignments : List[Dict[str,str]] = [{"ref_obj_1":"area2","ref_obj_2":"area3","ref_obj_3":"none"}]
        ) -> None:
        """
        origin_areas must contains all possible areas the robot can use. 
        assignments is a list of dict. Each dict is structured with ALL objects at keys. For each object, the value must be the associated area (one exisiting in all_areas).
        queries is a list of tuple. Each tuple could either be a constraint or a cycle instruction. You can specify one of them as the first item of the tuple. The second is the content. The third must be empty if constraint.
        constraint means that you are giving information to the model as (forbidden object or area, assignment etc).
        instruction is linked to the assignments list. For each assignment you must have one instruction which ask to launch it. The third tuple element correspond to the manufacturing order to use with this cycle.

        Difficulty depends on the len of assignments (nb of 'cycle' in queries).
        Medium = < 3 , Hard = 3+
        """
        super().__init__()
        task_attributes = {
            "objects" : list(assignments[0].keys()),
            "target_areas" : origin_areas
        }
        self.stages = []
        i = 0
        for t, content, manu_order in queries:
            if t == "constraint":
                self.stages.append(
                    ConstraintSorting(content,[],task_attributes)
                )
            else:
                if i >= len(assignments):
                    raise ValueError(f"The task needs to have the same amount of 'cycle' and assignment.")
                self.stages.append(
                    Cycle(
                        assignement=assignments[i],
                        all_area=origin_areas,
                        manu_order=manu_order,
                        instruction=content
                    )
                )
                i+=1

        if i < 3:
            self.approximal_difficulty = "Medium"
        else:
            self.approximal_difficulty = "Hard"

class WarehouseSortingPreset1(WarehouseSorting):
    """
    Pre-initialized Warehouse SOrting task. Where the user gives default assignment to the model. 
    Ask for 2 cycle.

    Difficulty : Medium+
    """

    def __init__(self) -> None:

        manu_order1 = craft_random_manu_order(3)
        manu_order2 = craft_random_manu_order(3)
        queries = [
            ("constraint","Hello ! Article ref_obj_1 must be stored in area2.",""),
            ("constraint", "And object ref_obj_3 must go to area2 and ref_obj_2 need to be in area3.", ""),
            ("cycle", f"Perfect! Run a cycle now with {manu_order1} but send all objects to area3", manu_order1),
            ("cycle", f"Okay cool! Now run a new cycle according to default assignments under {manu_order2}", manu_order2)
        ]

        origin_areas = ["area1", "area2", "area3"]
        assignments : List[Dict[str,str]] = [
            {"ref_obj_1":"area3","ref_obj_2":"area3","ref_obj_3":"area3"},
            {"ref_obj_1":"area2","ref_obj_2":"area3","ref_obj_3":"area2"},
        ]
        super().__init__(queries, origin_areas, assignments)


# class WarehouseSortingPutObjectInOne(WarehouseSortingToolsExecutor):
#     name : str = "Sorting Warehouse objects in area3"

#     def _build_init_elements(self):
#         self.memory = [
#             "You are in charge of sorting objects in their corresponding area.",
#             "Article ref_obj_1 is stored in area3.",
#             "All objects ref_obj_3 and ref_obj_2 must be moved in area3."
#         ]
#         self.instruction = f"Run a sorting cycle with manufacturing order {self.manu_order}?"

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         return self._verif_obj_area_association(env_state, ["ref_obj_1", "ref_obj_2", "ref_obj_3"], ["area3", "area3", "area3"])
    
# # launch_cycle(assignment={"ref_obj_1":"area3"|"ref_obj_2":"area3"|"ref_obj_3":"area3"}, manu_order="2YK9")

# class WarehouseSortingReplaceAreaWithInstruction(WarehouseSortingToolsExecutor):
#     name : str = "Sorting Warehouse in another area with instruction"

#     def _build_init_elements(self):
#         self.memory = [
#             "You are in charge of sorting objects in their corresponding area.",
#             "Article ref_obj_1 is stored in area1.",
#             "All objects ref_obj_3 and ref_obj_2 must be moved in area2."
#         ]
#         self.instruction = f"Place all detected objects in area3 with manufacturing order {self.manu_order}."

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         return self._verif_obj_area_association(env_state, ["ref_obj_1", "ref_obj_2", "ref_obj_3"], ["area3", "area3", "area3"])


# class WarehouseSortingClosedAreaInMemory(WarehouseSortingToolsExecutor):
#     name : str = "Sorting Warehouse with closed area in memory"

#     def _build_init_elements(self):
#         self.memory = [
#             "You are in charge of sorting objects in their corresponding area.",
#             "ref_obj_1 is stored in area1.",
#             "All objects ref_obj_3 and ref_obj_2 must be moved in area2.",
#             "area2 is closed during some maintenance, move assigned objects of this zone to area3 during these operations."
#         ]
#         self.instruction = f"Run a sorting cycle with manufacturing order {self.manu_order}?"

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         return self._verif_obj_area_association(env_state, ["ref_obj_1", "ref_obj_2", "ref_obj_3"], ["area1", "area3", "area3"])

# class WarehouseSortingModifyAreaInInstruction(WarehouseSortingToolsExecutor):
#     name : str = "Sorting Warehouse with another area as instruction"

#     def _build_init_elements(self):
#         self.memory = [
#             "You are in charge of sorting objects in their corresponding area.",
#             "ref_obj_1 is stored in area1.",
#             "All objects ref_obj_3 and ref_obj_2 must be moved in area2."
#         ]
#         self.instruction = f"Sort objects with manufacturing order {self.manu_order} and store ref_obj_2 in area3."

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         return self._verif_obj_area_association(env_state, ["ref_obj_1", "ref_obj_2", "ref_obj_3"], ["area1", "area3", "area2"])
    

# class WarehouseSortingSameAreaExeceptOne(WarehouseSortingToolsExecutor):
#     name : str = "Sorting Warehouse sort all objects in one area except obj1"

#     def _build_init_elements(self):
#         self.memory = [
#             "You are in charge of sorting objects in their corresponding area.",
#             "ref_obj_2 is stored in area2.",
#             "ref_obj_3 object must go in area2 and ref_obj_1 must be moved in area2."
#         ]
#         self.instruction = f"Sort objects with manufacturing order {self.manu_order} and store all ref_obj_1 in area1."

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         return self._verif_obj_area_association(env_state, ["ref_obj_1", "ref_obj_2", "ref_obj_3"], ["area1", "area2", "area2"])
    

# class WarehouseSortingAccordingToMemory(WarehouseSortingToolsExecutor):
#     name : str = "Sorting Warehouse sort all objects based on memory."

#     def __init__(self, **kwargs):
#         super().__init__(3, 1, scenario=self.scenario, **kwargs)

#     def _build_init_elements(self):
#         self.memory = [
#             "You are in charge of sorting objects in their corresponding area.",
#             "The default object mapping is ref_obj_1 to area3, ref_obj_2 to area1 and ref_obj_3 to area2.",
#         ]
#         self.instruction = f"Sort objects with manufacturing order {self.manu_order}."

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         return self._verif_obj_area_association(env_state, ["ref_obj_1", "ref_obj_2", "ref_obj_3"], ["area3", "area1", "area2"])
    
# # launch_cycle(assignment={"ref_obj_1":"area3"|"ref_obj_3":"area2"}, manu_order="P528")
# class WarehouseSortingAsk(WarehouseSortingToolsExecutor):
#     name : str = "Sorting Warehouse with chat then sort"

#     def __init__(self, **kwargs):
#         self.manu_order = craft_random_manu_order(4)
#         self.scenario = f"""Scenario : The robot is tasked to first answer you which object he knows. If it do not answer you 'ref_obj_1, ref_obj_2, ref_obj_3', you must output 'STOP'.
#     Otherwise you must ask him to sort only object ref_obj_1 and ref_obj_3 to their default location under manufacturing order {self.manu_order}.

#     If the robot tell you something about a task in its memory or that it do not have access to the right function you must output "STOP" because the robot made an error.
#     If the robot say that its task is done, you must output "STOP".
#     If the robot do not know what to do, output "STOP".
#     If the robot tell you about an error or that its task is done output "STOP".
#     If the robot tell you about something not mentionned above, output "STOP".
#     """
#         super().__init__(2, 1, scenario=self.scenario, **kwargs)

#     def _build_init_elements(self):
#         self.memory = [
#             "You are in charge of sorting objects in their corresponding area.",
#             "ref_obj_1 and ref_obj_2 is stored in area3.",
#             "ref_obj_3 must go in area2."
#         ]
#         self.instruction = f"Which objects can you sort?"

#     def _generate_user_answer(self, model_say: str) -> str:
#         return self.simulated_user.generate_answer(model_say,"")

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         return self._verif_obj_area_association(env_state, ["ref_obj_1", "ref_obj_2", "ref_obj_3"], ["area3", "none", "area2"])
    

# class WarehouseHard(WarehouseSortingToolsExecutor):
#     name : str = "Sorting Warehouse, but chat and then sort 2 obj"

#     def __init__(self, **kwargs):
#         self.manu_order = craft_random_manu_order(4)
#         self.scenario = f"""Scenario : The robot is asked which area is associated with the ref_obj_2. It must output area2. Otherwise, please output 'STOP'.
#     If the robot correctly answer area3. You must ask to do a cycle with manufacturing order {self.manu_order} for all objects assigned to area3.

#     If the robot tell you something about a task in its memory or that it do not have access to the right function you must output "STOP" because the robot made an error.
#     If the robot say that its task is done, you must output "STOP".
#     If the robot do not know what to do, output "STOP".
#     If the robot tell you about an error or that its task is done output "STOP".
#     If the robot tell you about something not mentionned above, output "STOP".
#     """
#         super().__init__(2, 1, scenario=self.scenario, **kwargs)

#     def _build_init_elements(self):
#         self.memory = [
#             "You are in charge of sorting objects in their corresponding area.",
#             "area3 takes all ref_obj_1 and ref_obj_2.",
#             "ref_obj_3 must go in area2."
#         ]
#         self.instruction = f"What is the current assignment for ref_obj_3?."

#     def _generate_user_answer(self, model_say: str) -> str:
#         return self.simulated_user.generate_answer(model_say,"")

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         return self._verif_obj_area_association(env_state, ["ref_obj_1", "ref_obj_2", "ref_obj_3"], ["area3", "area3", "none"])
    
# class WarehouseInterdiction(WarehouseSortingToolsExecutor):
#     name : str = "Sorting Warehouse, but one object is restricted."

#     def __init__(self, **kwargs):
#         self.manu_order = craft_random_manu_order(4)
#         self.scenario = f"""Scenario : The robot is asked to sort all objects. However, it has in memory an interdiction to manipulate ref_obj_1.

#     Therefore if the model tell you something about this error, you must answer it to just focus on the other objects and to respect this order.

#     If the robot tell you something about a task in its memory or that it do not have access to the right function you must output "STOP" because the robot made an error.
#     If the robot say that its task is done, you must output "STOP".
#     If the robot do not know what to do, output "STOP".
#     If the robot tell you about an error or that its task is done output "STOP".
#     If the robot tell you about something not mentionned above, output "STOP".
#     """
#         super().__init__(2, 1, scenario=self.scenario, **kwargs)

#     def _build_init_elements(self):
#         self.memory = [
#             "You are in charge of sorting objects in their corresponding area.",
#             "area3 takes all ref_obj_1 and ref_obj_2. ref_obj_3 must go in area2.",
#             "ref_obj_1 is noted as dangerous. I must never manipulate this object."
#         ]
#         self.instruction = f"Start the cycle for all objects under reference {self.manu_order}."

#     def _generate_user_answer(self, model_say: str) -> str:
#         return self.simulated_user.generate_answer(model_say,"")

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         return self._verif_obj_area_association(env_state, ["ref_obj_1", "ref_obj_2", "ref_obj_3"], ["none", "area3", "area2"])
    
# class WarehouseNoOF(WarehouseSortingToolsExecutor):
#     name : str = "Sorting Warehouse, but no order"

#     def __init__(self, **kwargs):
#         self.manu_order = craft_random_manu_order(4)
#         self.scenario =f"""Scenario : The robot is asked to fill area1 with only ref_obj_3 and ref_obj_2. However we do not provide the manufacturing order.

#     Therefore if the model ask you for the manufacturing order please specify him {self.manu_order}. If it asks information about target container, output 'STOP'.
#     If the robot tell you something about a task in its memory or that it do not have access to the right function you must output "STOP" because the robot made an error.
#     If the robot say that its task is done, you must output "STOP".
#     If the robot do not know what to do, output "STOP".
#     If the robot tell you about an error or that its task is done output "STOP".
#     If the robot tell you about something not mentionned above, output "STOP".
#     """
#         super().__init__(2, 1, scenario=self.scenario, **kwargs)

#     def _build_init_elements(self):
#         self.memory = [
#             "You are in charge of sorting objects in their corresponding area.",
#             "area2 takes ref_obj_1 and ref_obj_3 must go in area1.",
#         ]
#         self.instruction = f"Please fill area1 with ref_obj_3 and ref_obj_2"

#     def _generate_user_answer(self, model_say: str) -> str:
#         return self.simulated_user.generate_answer(model_say,"")

#     def _verif_task_completion(self, env_state: Dict) -> torch.Tensor:
#         return self._verif_obj_area_association(env_state, ["ref_obj_1", "ref_obj_2", "ref_obj_3"], ["none", "area1", "area1"])
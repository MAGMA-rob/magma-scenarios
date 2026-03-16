# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tasks import BaseTask
from magma_core.base.stage import AskingBaseStage
from magma_core.base.tasks_style import TaskStyle
from magma_core.base.data_structures import UserInstruction, EmptyInstruction

from typing import List, Dict, Tuple
import copy, random
from pathlib import Path

from .tools import WithoutManufacturingOrder
from .stages import Cycle, ConstraintSorting, ObjectToZone, AddLocationStage, RemoveLocationStage
from .att import OBJECTS, AREAS

# launch_cycle(assignment={"ref_obj_1":"area1"|"ref_obj_2":"area2"|"ref_obj_3":"area3"}, manu_order="A121")
# launch_cycle(assignment={"ref_obj_1":"area2"|"ref_obj_2":"area3"})
# launch_cycle(assignment={"ref_obj_1":"area2"|"ref_obj_3":"area3"})
# launch_cycle(assignment={"ref_obj_1":"area2"})

class NoManuPreset(BaseTask):
    """
    Class to create a Sorting task in a factory. Robot knows some objects and area. Users gives constraint about objects assignment.
    The robot must solves these constraint to complete multiple cycle.
    """
    name : str = "Parent Sorting Warehouse objects"
    env_id : str = "SortingCubesWarehouse-v1"

    Tools_cls = WithoutManufacturingOrder

    styles = [
        TaskStyle.CONSTRAINED,
        TaskStyle.LONG_STAGE
    ]

    randomized_config_path = str(Path(__file__).parent.joinpath("config.yaml"))

    all_task_attributes = {
        "objects" : OBJECTS,
        "target_areas" : AREAS
    }

    def __init__(
            self,
            queries : List[Tuple] = [
                ("constraint","Hello do not forget that ref_obj_1 goes to area2, ref_obj_2 goes to area3 and ref_obj_3 goes to area1"),
                ("constraint","Hey, ref_obj_3 must not be used anymore."),
                ("cycle","Can you launch a cycle according to your memory?")
                ],
            nb_of_object : int = 3,
            nb_of_area : int = 3,
            assignments : List[Dict[str,str]] = [{"ref_obj_1":"area2","ref_obj_2":"area3","ref_obj_3":"none"}]
        ) -> None:
        """
        origin_areas must contains all possible areas the robot can use. 
        assignments is a list of dict. Each dict is structured with ALL objects at keys. For each object, the value must be the associated area (one exisiting in all_areas).
        queries is a list of tuple. Each tuple could either be a constraint, a question or a cycle instruction. You can specify one of them as the first item of the tuple. The second is the content.
        constraint means that you are giving information to the model as (forbidden object or area, assignment etc).
        instruction is linked to the assignments list. For each assignment you must have one instruction which ask to launch it.
        question means that you are asking a question to the user. In that specific case, you must past 3 items. First one is set to 'question', the second is the content of the question and the third, what the model must answer.

        Difficulty depends on the len of assignments (nb of 'cycle' in queries).
        Medium = < 3 , Hard = 3+
        """
        super().__init__()

        if nb_of_object > len(OBJECTS) or nb_of_area > len(AREAS) or nb_of_object <= 0 or nb_of_area <=0:
            raise TypeError(f"Non Valid number of object ({nb_of_object}) or number of area ({nb_of_area}) passed.")

        known_objects = OBJECTS[:nb_of_object]
        known_areas = AREAS[:nb_of_area]
        task_attributes = {
            "objects" : known_objects,
            "target_areas" : known_areas
        }
        
        self.stages = []
        i=0
        for tupl in queries:
            t, content = tupl[0], tupl[1]
            if t == "constraint":
                self.stages.append(
                    ConstraintSorting(content,[],copy.deepcopy(task_attributes))
                )
            elif t == "cycle" or t == "cycle-flag":
                if i >= len(assignments):
                    raise ValueError(f"The task needs to have the same amount of 'cycle' and assignment.")
                if content == "none":
                    ins = EmptyInstruction()
                else:
                    ins = UserInstruction(content)
                self.stages.append(
                    Cycle(
                        assignment=assignments[i],
                        instruction=ins,
                        known_areas=task_attributes["target_areas"].copy(),
                        flag_answer= "flag" in t
                    )
                )
                i+=1
            elif t == "uni":
                if i >= len(assignments):
                    raise ValueError(f"The task needs to have the same amount of 'cycle' and assignment.")
                self.stages.append(ObjectToZone(assignments[i], task_attributes["objects"].copy(), content))
                i+=1
            else:
                if len(tupl) < 2:
                    raise TypeError(f"Not enought tuple element for type {t} : {tupl}")
                if t == "question":
                    self.stages.append(AskingBaseStage(content,tupl[2],[],copy.deepcopy(task_attributes)))
                elif t == "add" or t == "add-noflag":
                    if isinstance(tupl[2], str):
                        self.stages.append(AddLocationStage(
                            UserInstruction(content),
                            tupl[2],
                            [],
                            copy.deepcopy(task_attributes),
                            flag_answer_to_user= not "noflag" in t
                        ))
                        task_attributes["target_areas"].append(tupl[2])
                    elif isinstance(tupl[2], List):
                        ins = UserInstruction(content)
                        for j, area in enumerate(tupl[2]):
                            self.stages.append(AddLocationStage(
                                ins,
                                area,
                                [],
                                copy.deepcopy(task_attributes),
                                flag_answer_to_user= j == len(area)-1
                            ))
                            ins = EmptyInstruction()
                            task_attributes["target_areas"].append(area)
                    else:
                        raise TypeError(f"Unknow value (wanted either str or a list of str) : {tupl[2]} with type {type(tupl[2])}")
                        
                elif t == "remove" or t == "remove-noflag":
                    if isinstance(tupl[2], str):
                        self.stages.append(RemoveLocationStage(
                            UserInstruction(content),
                            tupl[2],
                            [],
                            copy.deepcopy(task_attributes),
                            flag_answer_to_user=not "noflag" in t
                        ))
                        task_attributes["target_areas"].remove(tupl[2])
                    elif isinstance(tupl[2], List):
                        ins = UserInstruction(content)
                        for j, area in enumerate(tupl[2]):
                            self.stages.append(RemoveLocationStage(
                                ins,
                                area,
                                [],
                                copy.deepcopy(task_attributes),
                                flag_answer_to_user= j == len(area)-1
                            ))
                            ins = EmptyInstruction()
                            task_attributes["target_areas"].remove(area)
                    else:
                        raise TypeError(f"Unknow value (wanted either str or a list of str) : {tupl[2]} with type {type(tupl[2])}")
                
                else:
                    raise TypeError(f"Unknow type {t} for {tupl}. Known types are uni, question, constraint, cycle")

        if i < 3:
            self.approximal_difficulty = "Medium"
        else:
            self.approximal_difficulty = "Hard"

class WarehouseSortingSimpPreset1(NoManuPreset):
    """
    A preset for WarehouseSortingSimp Task.
    Here we ask the model if it knows some default assignment (answer is no). We give it some assignment then ask for a cycle.
    If you set difficulty to 1, we will modify the constraint and try to relaunch a novel cycle.
    """

    def __init__(self, difficulty : int = 0):
        queries : List[Tuple] = [
            ("question","Hello machine, do you know where ref_obj_1 must go?", "The model must answer that it does not have any assignment with ref_obj_1, or with any objects."),
            ("constraint","Okay, today cycles must send all objects to area2."),
            ("uni","Good, let's make a test. Can you store only one ref_obj_1 in its area without using the cycle method?"),
            ("cycle","Perfect! Launch a cycle now for all objects"),
            ]
        assignments : List[Dict[str,str]] = [
            {"ref_obj_1":"area2"},
            {"ref_obj_1":"area2","ref_obj_2":"area2","ref_obj_3":"area2"}
        ]
        if difficulty >= 1:
            queries.extend([
                ("constraint", "Hey, we have new rules! ref_obj_2 is going to area1 now"),
                ("cycle", "Please relaunch a cycle now")
            ])
            assignments.append({"ref_obj_1":"area2","ref_obj_2":"area1","ref_obj_3":"area2"})

        super().__init__(queries, 3, 3, assignments)

class WarehouseSortingSimpPreset2(NoManuPreset):
    """
    A preset for WarehouseSortingSimp Task.
    Here we give to the model full default assignment. Followed by a cycle. Then we present it a new area it can use for some objects.
    Then launch a cycle using this new area.
    You can select a difficulty from 0 to 3, adding some random cycle or sorting instruction in between.
    """
    
    def __init__(self, difficulty : int = 1) -> None:
        obj = OBJECTS[:3].copy()
        random.shuffle(obj)

        queries : List[Tuple] = [
            ("constraint", f"Hello, please consider {AREAS[0]} the default for {obj[0]} and {obj[1]}."),
            ("cycle-flag", "Launch a cycle for these objects only"),
        ]
        assignments = [{obj[0]:AREAS[0],obj[1]:AREAS[0]}]

        for i in range(difficulty):
            queries.append(("cycle-flag", f"Hey! You can launch a cycle for {obj[0]} and {obj[1]}"))
            assignments.append({obj[0]:AREAS[0],obj[1]:AREAS[0]})

        queries.extend([
            ("question", "Hello, do you have an object which do not have any assignment?", f"The model must answer that {obj[2]}"),
            ("add", f"Hey, you can use this new area4 for {obj[2]}. Add it to your system","area4"),
            ("cycle", "You can launch a cycle for all objects")
        ])

        assignments.append({obj[0]:AREAS[0],obj[1]:AREAS[0],obj[2]:AREAS[1]})
        
        super().__init__(queries, 3, 2, assignments)

class WarehouseSortingSimpPreset3(NoManuPreset):
    """
    A preset for WarehouseSortingSimp Task.
    Here we give a simple constraint and do two cycle. We repeat this n times depending on the difficulty you choose
    """

    def __init__(self, difficulty : int = 3):
        areas = random.sample(self.all_task_attributes["target_areas"],k=2)
        queries : List[Tuple] = [
            ("constraint",f"Okay, consider {areas[0]} the default assignment for ref_obj_1 and {areas[1]} for ref_obj_2 and ref_obj_3."),
            ("cycle-flag","Perfect! Launch a cycle now for all objects"),
            ]
        base = {"ref_obj_1":areas[0],"ref_obj_2":areas[1],"ref_obj_3":areas[1]}
        assignments : List[Dict[str,str]] = [
            base.copy()
        ]
        for _ in range(difficulty):
            area = random.choice(self.all_task_attributes["target_areas"])
            obj = random.choice(self.all_task_attributes["objects"])
            queries.extend([
                ("constraint",f"Hey! the target area for {obj} has changed! It's {area} now."),
                ("cycle-flag","Hello, you can launch a cycle for all objects!"),
            ])
            base[obj] = area
            assignments.append(base.copy())

        super().__init__(queries, 3, 5, assignments)

class WarehouseSortingSimpPreset4(NoManuPreset):
    """
    A preset for WarehouseSortingSimp Task.
    Here we give assignment then cycle. Then we give a constraint about that future cycle concern only object associated to a specific area.
    Then do cycle. Difficulty increase the number of cycle we are doing and asking for multiple cycle.
    """

    def __init__(self, difficulty : int = 0):
        areas = random.sample(self.all_task_attributes["target_areas"],k=2)
        queries : List[Tuple] = [
            ("constraint",f"For future cycle, only objects that are associated with {areas[1]} are concerned okay?"),
            ("constraint",f"Okay, consider {areas[0]} the default assignment for ref_obj_1 and {areas[1]} for ref_obj_2 and ref_obj_3."),
            ("cycle-flag",f"Please launch a cycle right now please"),
            ("cycle-flag",f"I need a cycle now"),
            ]
        assignments : List[Dict[str,str]] = [
            {"ref_obj_2":areas[1],"ref_obj_3":areas[1]},
            {"ref_obj_2":areas[1],"ref_obj_3":areas[1]}
        ]
        for _ in range(difficulty):
            ar = random.choice(self.all_task_attributes["target_areas"])
            queries.extend([
                ("cycle-flag",f"Just for this time, I need a cycle for ref_obj_1 and ref_obj_2 to {ar} please"),
                ("cycle-flag",f"Please launch a cycle right now please"),
            ])
            assignments.extend([
                {"ref_obj_1":ar,"ref_obj_2":ar},
                {"ref_obj_2":areas[1],"ref_obj_3":areas[1]}
            ])

        super().__init__(queries, 3, 5, assignments)

class WarehouseSortingSimpPreset5(NoManuPreset):
    """
    A preset for WarehouseSortingSimp Task.
    Here we class objects in two category. Each category have a target. Then we ask for multiple cycle.
    If difficulty is >=1, we randomly change some target every two cycle request.
    """
    
    def __init__(self, difficulty : int = 0, cycle_request : int = 4, multi_steps : bool = False) -> None:
        obj = OBJECTS[:3].copy()
        random.shuffle(obj)
        areas = random.sample(self.all_task_attributes["target_areas"],k=2)
 
        queries : List[Tuple] = [
            ("constraint", f"By the way, drinks are going to {areas[0]} and food to {areas[1]}"),
            ("constraint", f"Hello, please consider {obj[0]} and {obj[1]} as drinks. {obj[2]} is food"),
        ]
        assignments = []

        for i in range(cycle_request//2):
            a = [
                {obj[0]:areas[0],obj[1]:areas[0]},
                {obj[2]:areas[1]}
            ]
            if random.randint(0,1)>=1:
                queries.extend([
                    ("cycle","Hello, I need a cycle for drinks and right after for food!"),
                    ("cycle-flag","none")
                ])
                assignments.extend(a)
            else:
                idx = [0,1]
                random.shuffle(idx)
                s = [
                    ("cycle-flag", f"Hey! You can launch a drinks cycle please"),
                    ("cycle-flag", f"Do a cycle for food now"),
                ]
                queries.extend([
                        s[idx[0]],
                        s[idx[1]]
                    ])
                assignments.extend([
                    a[idx[0]],
                    a[idx[1]]
                ])
            if difficulty >=1:
                areas = random.sample(self.all_task_attributes["target_areas"],k=2)
                queries.append(("constraint",f"From now, drinks are going to {areas[0]} and food to {areas[1]}"))
        
        super().__init__(queries, 3, 5, assignments)

class WarehouseSortingSimpPreset6(NoManuPreset):
    """
    A preset for WarehouseSortingSimp Task.
    Here we ask the model to do some take and depose while updating constraint.
    Difficulty is the number of repetition of this schema
    """
    # launch_cycle(assignment={"ref_obj_1":"area4"|"ref_obj_2":"area1"})
    def __init__(self, difficulty : int = 3)-> None:

        nb_of_object: int = 3
        nb_of_area: int = 4

        areas = random.sample(self.all_task_attributes["target_areas"][:nb_of_area],k=2)

        ori = {"ref_obj_1":areas[0],"ref_obj_2":areas[1],"ref_obj_3":areas[0]}

        queries: List[Tuple] = [
            ("constraint", f"Consider {areas[0]} the default assignement for ref_obj_1 and ref_obj_3. {areas[1]} is for the rest."), 
            ("uni", f"I want you to take ref_obj_2 and place it in its target area?"),
            ]
        assignments: List[Dict[str, str]] = [{"ref_obj_2": areas[1]}]

        for _ in range(difficulty):
            
            obj = random.choice(self.all_task_attributes["objects"][:nb_of_object])

            queries.extend([ 
                ("uni", f"I want you to take a {obj} and put in {ori[obj]} whitout using the cycle method."),
                ("question", "Perfect, Now wait for further instruction", "The model must acknowledge and do nothing")
            ])
            assignments.append({obj:ori[obj]})

        super().__init__(queries, nb_of_object, nb_of_area, assignments)

class WarehouseSortingSimpPreset7(NoManuPreset):
    """
    A preset for WarehouseSortingSimp Task.
    Here we ask the model to do some cycle, then we give constraint, ask question etc..
    """
    # launch_cycle(assignment={"ref_obj_1":"area4"|"ref_obj_2":"area1"})
    def __init__(self, difficulty : int = 3)-> None:

        nb_of_object: int = 3
        nb_of_area: int = 4

        areas = random.sample(self.all_task_attributes["target_areas"][:nb_of_area],k=2)

        ori = {"ref_obj_1":areas[0],"ref_obj_2":areas[1],"ref_obj_3":areas[0]}

        queries: List[Tuple] = [
            ("constraint", f"Consider {areas[0]} the default assignement for ref_obj_1 and ref_obj_3. {areas[1]} is for the rest."), 
            ("cycle-flag", f"Can you sort all ref_obj_2 and ref_obj_1 to their target area?"),
            ]
        assignments: List[Dict[str, str]] = [{"ref_obj_2": areas[1], "ref_obj_1": areas[0]}]

        for _ in range(difficulty):
            obj = random.choice(self.all_task_attributes["objects"][:nb_of_object])
            ar = random.choice(areas)
            queries.extend([ 
                ("question", f"Hey robot, do you know where should you put {obj}", f"The model must answer that {obj} goes to {ori[obj]}"),
                ("cycle-flag", f"Hey! I want you to to launch a cycle for all objects. However, you must not use {ar} anymore. All objects must temporary goes to {AREAS[3]}"),
                ("question", f"Can you launch a cycle for {obj} to {ar} please?", f"Model must answer that {ar} can not be used anymore"),
                ("cycle-flag", f"{ar} is fine now. I will test it. Launch a cycle for {obj} in {ar} only please.")
            ])
            assignments.extend([{"ref_obj_2": AREAS[3], "ref_obj_1": AREAS[3], "ref_obj_3": AREAS[3]},{obj:ar}])

        super().__init__(queries, nb_of_object, nb_of_area, assignments)

class WarehouseSortingSimpAdd(NoManuPreset):
    """
    A preset for WarehouseSortingSimp Task.
    It allows to verify that the model is able to add a new area and remove older ones.
    """
    # launch_cycle(assignment={"ref_obj_1":"area4"|"ref_obj_2":"area1"})
    def __init__(self)-> None:      
        queries: List[Tuple] = [
            ("remove", "Hello we won't use anymore the area2 you can remove it!","area2"), 
            ("add", "But here is a new area, called area4 to use for ref_obj_1","area4"), 
            ("cycle", "Can you launch a cycle for all objects you know. If an object have no default assignment, send it to area1")]
        nb_of_object: int = 2
        nb_of_area: int = 2
        assignments: List[Dict[str, str]] = [{"ref_obj_1": "area4","ref_obj_2": "area1"}] 
        super().__init__(queries, nb_of_object, nb_of_area, assignments)

class WarehouseSortingSimpAdd2(NoManuPreset):
    """
    A preset for WarehouseSortingSimp Task.
    It allows to verify that the model is able to add multiples new areas while maintening memory consistensy of informations.
    """
    # launch_cycle(assignment={"ref_obj_1":"area4"|"ref_obj_2":"area1"})
    def __init__(self)-> None:      
        queries: List[Tuple] = [
            ("add", "Can you add area4 and area5 to your known areas please?",["area4","area5"]), 
            ("constraint", "area4 is for ref_obj_1 and area5 is for the rest."), 
            ("cycle", "Can you launch a cycle for all objects please?")
            ]
        nb_of_object: int = 3
        nb_of_area: int = 2
        assignments: List[Dict[str, str]] = [{"ref_obj_1": "area4","ref_obj_2": "area5", "ref_obj_3": "area5"}] 
        super().__init__(queries, nb_of_object, nb_of_area, assignments)

class WarehouseSortingSimpAdd3(NoManuPreset):
    """
    A preset for WarehouseSortingSimp Task.
    We are asking the model to remove area, then to launch a cycle using a removed areas. To verify consistensy.
    """
    # launch_cycle(assignment={"ref_obj_1":"area4"|"ref_obj_2":"area1"})
    def __init__(self)-> None:
        areas = AREAS[:5].copy()
        random.shuffle(areas)
        queries: List[Tuple] = [
            ("remove", f"Can you remove {areas[0]} from your known area and consider {areas[4]} as the default target for all objects", areas[0]), 
            ]
        assignments: List[Dict[str, str]] = []

        for i in range(3):
            queries.extend([
                ("question",f"Can you launch a cycle for obj_ref_1 to {areas[i]}, and other objects to the default one?", f"The model must refuse because {areas[i]} does not exist anymore."),
                ("cycle-flag",f"Ok, send all objects to the default one then"),
                ("remove",f"Hello robot, the {areas[i+1]} can not be used anymore. You can remove it.", areas[i+1]),
            ])
            assignments.append({"ref_obj_1":areas[4],"ref_obj_2":areas[4],"ref_obj_3":areas[4]})

        super().__init__(queries, 3, 5, assignments)

class WarehouseSortingSimpInterrupt(NoManuPreset):
    """
    A preset for WarehouseSortingSimp Task.
    The idea is to ask the model to do a cycle according default assignment, then to interupt it with a new cycle to do and ask him to continue its precedent task after.
    """
    # launch_cycle(assignment={"ref_obj_1":"area4"|"ref_obj_2":"area1"})
    def __init__(self, difficulty : int = 0)-> None:
        nb_of_object: int = 2
        nb_of_area: int = 3
        areas = random.sample(self.all_task_attributes["target_areas"][:nb_of_area],k=2)
        queries: List[Tuple] = [
            ("constraint", f"Consider {areas[0]} the default area for ref_obj_1 and {areas[1]} for ref_obj_2"), 
            ("cycle", "Can you launch a cycle for all objects please?"),
        ]
        assignments: List[Dict[str, str]] = [{"ref_obj_1": areas[0],"ref_obj_2": areas[1]}]

        if difficulty >=1:
            queries.extend([
                ("add-noflag", 
                 "Robot, stop what you are doing. I need you to add area4 and right after, send to it all ref_obj_1! When you have done that, you can continue your precedent task.",
                 "area4"),
                ("cycle", "none")
            ])
            assignments.append(
                {"ref_obj_1": "area4"}
            )
        else:
            queries.extend([("cycle", f"Hey, stop what you are doing I need you to launch a cycle for ref_obj_1 to {areas[1]} right now. You will continue your task after.")])
            assignments.append(
                {"ref_obj_1":areas[1]}
            )

        queries.append(("cycle-flag", "none"))
        assignments.append({"ref_obj_1": areas[0],"ref_obj_2": areas[1]})
        super().__init__(queries, nb_of_object, nb_of_area, assignments)

class WarehouseSortingSimpInterrupt2(NoManuPreset):
    """
    A preset for WarehouseSortingSimp Task.
    The idea is to ask the model to do a cycle according default assignment, then to interupt it with a new cycle to do and ask him to continue its precedent task after.
    """
    # launch_cycle(assignment={"ref_obj_1":"area4"|"ref_obj_2":"area1"})
    def __init__(self)-> None:
        nb_of_object: int = 3
        nb_of_area: int = 5
        areas = random.sample(self.all_task_attributes["target_areas"][:nb_of_area],k=2)
        ori = {"ref_obj_1":areas[0],"ref_obj_2":areas[1],"ref_obj_3":areas[2]}
        obj = random.sample(list(ori.keys()),k=2)
        queries: List[Tuple] = [
            ("constraint", f"Consider {ori[obj[0]]} the default area for {obj[0]} and {ori[obj[1]]} for {obj[1]}"), 
            ("cycle", f"Can you launch a cycle for {obj[0]} and {obj[1]} objects please?"),
            ("cycle", f"Hey, stop what you are doing I need you to launch a cycle for ref_obj_1 to {areas[3]} right now. You will continue your task after."),
            ("cycle-flag", "none"),
            ("question", f"Do you know where {obj[2]} must go?", "The model must answer no"),
            ("cycle", f"It must go to {ori[obj[2]]}. Launch a cycle directly for all objects with this update."),
            ("cycle", f"Hey, stop what you are doing I need you to launch a cycle for ref_obj_2 to {areas[4]} and ref_obj_1 to {areas[3]} right now. You will continue your task after."),
            ("cycle-flag", "none"),
        ]
        assignments: List[Dict[str, str]] = [
            {obj[0]: ori[obj[0]],obj[1]: ori[obj[1]]},
            {"ref_obj_1":areas[3]},
            {obj[0]: ori[obj[0]],obj[1]: ori[obj[1]]},
            {obj[0]: ori[obj[0]],obj[1]: ori[obj[1]],obj[2]: ori[obj[2]]},
            {"ref_obj_1":areas[3],"ref_obj_2":areas[4]},
        ]
        super().__init__(queries, nb_of_object, nb_of_area, assignments)

class WarehouseSortingSimpInterdictionPreset1(NoManuPreset):
    """
    A preset for WarehouseSortingSimp Task.
    In this preset, we are passing restritive constraint to the model. Then ask the model to do cycle which break these constraint. We are testing the capicity of the model to refuse.
    """

    def __init__(self, difficulty : int = 0)-> None:
        nb_of_object: int = 2
        nb_of_area: int = 2
        areas = random.sample(self.all_task_attributes["target_areas"][:nb_of_area],k=2)
        queries: List[Tuple] = [
            ("constraint", f"The default cycle rule is {areas[0]} for ref_obj_1 and {areas[1]} for ref_obj_2"),  
            ("constraint", "However, do not manipulate yet ref_obj_1. I am fixing somme issues with this type of object."),
            ("question", "Can you launch a cycle for all objects please?", "The robot must refuses due to the fact that ref_obj_1 must not be manipulated."),
            ("cycle-flag", "Oh okay, then launch a cycle just for ref_obj_2")
        ]
        assignments: List[Dict[str, str]] = [{"ref_obj_2": areas[1]}]

        if difficulty >=1:
            queries.extend([
                ("question", "Hello, I need a cycle for all objects right now?", "The robot must refuses due to the fact that ref_obj_1 must not be manipulated."),
                ("cycle-flag", "Oh okay, then launch a cycle just for ref_obj_2"),
                ("cycle-flag", f"Hello, Launch a cycle for ref_obj_2 only but to {areas[0]} this time")
            ])
            assignments.extend([
                {"ref_obj_2": areas[1]},
                {"ref_obj_2": areas[0]}
            ])
        
        queries.append(("cycle","It's okay now. ref_obj_1 are fixed. Launch a cycle for all objects according to their default assignment!"))
        assignments.append({"ref_obj_1": areas[0],"ref_obj_2": areas[1]})
        super().__init__(queries, nb_of_object, nb_of_area, assignments)

class WarehouseSortingSimpInterdictionPreset2(NoManuPreset):
    """
    A preset for WarehouseSortingSimp Task.
    In this preset, we are passing restritive constraint to the model. Then ask the model to do cycle which break these constraint. We are testing the capicity of the model to refuse.
    Difficulty allows to repeat the deletion of constraints and the add of a new ones.
    """

    def __init__(self, difficulty : int = 0)-> None:
        nb_of_object: int = 2
        nb_of_area: int = 4
        areas = random.sample(self.all_task_attributes["target_areas"][:nb_of_area],k=2)
        queries: List[Tuple] = [
            ("constraint", f"Consider {areas[0]} the default area for ref_obj_1 and {areas[1]} for ref_obj_2"), 
            ("constraint", f"However, you should not use {areas[0]}."),
            ("question", "Can you launch a cycle for all objects please to their default areas?", f"The robot must refuses due to the fact that {areas[0]} must not be manipulated."),
            ("cycle-flag", "Oh okay, then launch a cycle just for ref_obj_2")
        ]
        
        assignments: List[Dict[str, str]] = [{"ref_obj_2": areas[1]}]

        if difficulty >= 1:
            queries.extend([
                ("question", f"Hello, can you send all objects to {areas[0]}?", f"The robot must refuses due to the fact that {areas[0]} must not be manipulated."),
                ("question", f"Do you have another area avalaible?", f"The robot must indicates any area other than {areas[0]}. Acceptable values : {areas[1:]}. If the robot indicates at least one, it's valid."),
                ("cycle-flag", f"Hey {areas[0]} is not in maintenance anymore. Launch a cycle for all objects according to their default location"),
            ])
            assignments.extend([
                {"ref_obj_2": areas[1],"ref_obj_1": areas[0]},
            ])
        
        super().__init__(queries, nb_of_object, nb_of_area, assignments)
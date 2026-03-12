from typing import Dict, Any, List
import random

from magma_core.base.stage import BaseTaskStage
from magma_core.base.user_request import BaseRequest
from magma_core.base.state import TaskState
from magma_core.base.data_structures import UserInstruction

from .stages import Cycle, ObjectToZone
from magma_scenarios.templates.stages import MissingInformationStage, ForbiddenElemStage
from magma_scenarios.templates.constraints import ObjectAssignmentConstraint

class CycleRequest(BaseRequest):

    def __init__(self, max_object_per_cycle_request : int = 3) -> None:
        super().__init__()
        self.max_object = max_object_per_cycle_request

    def _create_stages(
            self,
            objects_to_sort : List,
            all_areas : List,
            state : TaskState,
            base_assignement : Dict = {}
        ) -> List[BaseTaskStage]:
        stages = []

        assignement = base_assignement.copy()
        missing_assignment = {}
        forbidden_object = []
        obj_with_forbidden_zones = []

        for obj in objects_to_sort:
            # 1 : Verify that the object is not forbidden
            if obj in state.properties.get("forbidden_objects",[]):
                forbidden_object.append(obj)
            
            # 2 : Verify if the object have a default assignement (if we do not give it as a base assignment)
            if not obj in assignement:
                target_area = state.relations.get("object_zone",{}).get(obj, "none")
                if target_area not in all_areas:
                    target_area = random.choice(all_areas)
                    missing_assignment[obj] = target_area
                    # here we continue for the forbidden zone
                else:
                    assignement[obj] = target_area
            else:
                target_area = assignement[obj]
            
            # 3 : Forbidden zone
            if target_area in state.properties.get("forbidden_zones", []):
                obj_with_forbidden_zones.append(obj)

        assignement.update(missing_assignment)
        objs = " and ".join(assignement.keys())
        instruction_str = f"Launch a cycle for {objs}."
        if base_assignement:
            instruction_str += " And consider "
            for obj, area in base_assignement.items():
                instruction_str += f"{obj} to {area}"
            instruction_str += " as news default assignment."
        cycle_instruction = UserInstruction(instruction_str)

        if obj_with_forbidden_zones or forbidden_object:
            objs = " and ".join(forbidden_object)
            areas = " and ".join([assignement[o] for o in obj_with_forbidden_zones])

            answer = "The model must inform that "
            if forbidden_object:
                answer+= f"{objs} are forbidden"
            if obj_with_forbidden_zones:
                answer+= f"{areas} can not be used."

            s = ForbiddenElemStage(
                instruction=cycle_instruction,
                answer=answer,
                memory=[],
                attributes=state.attributes
            )
            stages.append(
                s
            )

            # TO DO: Random override (to keep rules respect some times)
            cycle_instruction = UserInstruction("Please override these orders just for my cycle")

        if missing_assignment:
            objs = " and ".join(missing_assignment.keys())
            stages.append(MissingInformationStage(
                instruction=cycle_instruction,
                answer=f"The robot must ask about target areas for {objs}",
                memory=[],
                attributes=state.attributes
            ))
            ins = "For this cycle, "
            for o, a in missing_assignment.items():
                ins += f"{o} goes to {a}, "
            cycle_instruction = UserInstruction(ins)
        
        stages.append(Cycle(
            assignment=assignement,
            known_areas=all_areas,
            flag_answer=True,
            instruction=cycle_instruction
        ))

        return stages


    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        all_objects = state.entities.get("objects", []).copy()
        all_areas = state.entities.get("zones", [])

        if len(all_objects) <= 0 or len(all_areas) <=0:
            raise RuntimeError(f"Failed to build the stage from {self.__class__.__name__} due to empty objects or areas")
        
        nb_obj = random.randint(1,min(self.max_object, len(all_objects)))
        random.shuffle(all_objects)

        return self._create_stages(all_objects[:nb_obj], all_areas, state)

class CycleWithPermanentRulesRequest(CycleRequest):

    constraints : list[ObjectAssignmentConstraint]

    def __init__(self, max_object_per_cycle_request: int = 3, max_permanent_rule : int = 2) -> None:
        super().__init__(max_object_per_cycle_request)
        self.max_rules = max_permanent_rule

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        all_objects = state.entities.get("objects", []).copy()
        all_areas = state.entities.get("zones", [])
        self.constraints = []

        if len(all_objects) <= 0 or len(all_areas) <=0:
            raise RuntimeError(f"Failed to build the stage from {self.__class__.__name__} due to empty objects or areas")
        
        nb_obj = random.randint(1,min(self.max_object, len(all_objects)))
        random.shuffle(all_objects)

        nb_rules = random.randint(1,min(nb_obj,self.max_rules))

        assignement = {}
        for i in range(nb_rules):
            a = random.choice(all_areas)
            assignement[all_objects[i]] = a
            self.constraints.append(ObjectAssignmentConstraint(all_objects[i], a))
        
        
        return self._create_stages(all_objects[:nb_obj], all_areas, state, base_assignement=assignement)
    
    def apply_request(self, state: TaskState) -> TaskState:
        for c in self.constraints:
            c.apply(state)
        return state
        

class MoveOneObjectRequest(BaseRequest):

    def __init__(self) -> None:
        super().__init__()

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:        
        all_objects = state.entities.get("objects", [])
        all_areas = state.entities.get("zones", [])

        if len(all_objects) <= 0 or len(all_areas) <=0:
            raise RuntimeError(f"Failed to build the stage from {self.__class__.__name__} due to empty objects or areas")
        
        obj = random.choice(all_objects)
        area = random.choice(all_areas)

        return [
            ObjectToZone(
                {obj: area},
                all_objects,
                f"Can you store one {obj} to {area} without using your cycle mode"
            )
        ]
        
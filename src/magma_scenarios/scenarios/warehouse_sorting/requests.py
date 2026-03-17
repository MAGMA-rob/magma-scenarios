from typing import Dict, Any, List
import random
from copy import deepcopy

from magma_core.base.stage import BaseTaskStage, ModifAttributesBaseStage
from magma_core.base.user_request import BaseRequest
from magma_core.base.state import TaskState
from magma_core.base.data_structures import UserInstruction, EmptyInstruction

from .stages import ObjectToZone
from magma_scenarios.templates.stages import MissingInformationStage, ForbiddenElemStage, Cycle
from magma_scenarios.templates.constraints import ObjectAssignmentConstraint
from magma_scenarios.templates.requests import AddValueToListRequest, RemoveValueToListRequest

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
        obj_with_forbidden_areas = []

        for obj in objects_to_sort:
            # 1 : Verify that the object is not forbidden
            if obj in state.properties.get("forbidden_objects",[]):
                forbidden_object.append(obj)
            
            # 2 : Verify if the object have a default assignement (if we do not give it as a base assignment)
            if not obj in assignement:
                target_area = state.relations.get("object_area",{}).get(obj, "none")
                if target_area not in all_areas:
                    target_area = random.choice(all_areas)
                    missing_assignment[obj] = target_area
                    # here we continue for the forbidden zone
                else:
                    assignement[obj] = target_area
            else:
                target_area = assignement[obj]
            
            # 3 : Forbidden zone
            if target_area in state.properties.get("forbidden_areas", []):
                obj_with_forbidden_areas.append(obj)

        assignement.update(missing_assignment)
        objs = " and ".join(assignement.keys())
        instruction_str = f"Launch a cycle for {objs}."
        if base_assignement:
            instruction_str += " And consider "
            for obj, area in base_assignement.items():
                instruction_str += f"{obj} to {area}"
            instruction_str += " as news default assignment."
        cycle_instruction = UserInstruction(instruction_str)

        if obj_with_forbidden_areas or forbidden_object:
            objs = " and ".join(forbidden_object)
            areas = " and ".join([assignement[o] for o in obj_with_forbidden_areas])

            answer = "The model must inform that "
            if forbidden_object:
                answer+= f"{objs} are forbidden"
            if obj_with_forbidden_areas:
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
        all_objects = state.attributes.get("objects", []).copy()
        all_areas = state.attributes.get("target_areas", [])

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
        all_objects = state.attributes.get("objects", []).copy()
        all_areas = state.attributes.get("target_areas", [])
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
        all_objects = state.attributes.get("objects", [])
        all_areas = state.attributes.get("target_areas", [])

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
    
class CycleByCategoriesRequest(CycleRequest):

    def __init__(self, max_categories_per_cycle_request : int = 2) -> None:
        super().__init__()
        self.max_categories = max_categories_per_cycle_request

    # On tire au piff parmis les areas qui existent. Soit elles ont des objets et on fait, soit 
    # On explique qu'elles ne sont attribuées à aucune

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        all_objects = state.relations.get("objects", [])
        all_areas = state.attributes.get("target_areas", [])

        if len(all_objects) <=0:
            raise RuntimeError(f"Failed to build the stage from {self.__class__.__name__} due to empty objects or areas")
        
        nb_cat = random.randint(1,min(self.max_categories, len(all_objects)))

        known_types = list(state.relations.keys())
        
        for _, type in state.relations["object_type"].items():
            if type not in known_types:
                known_types.append(type)

        # ca va pas marcher.

        return self._create_stages(all_objects[:nb_obj], all_areas, state)

class AddAreas(AddValueToListRequest):

    def __init__(self, all_areas : List[str], max_update : int = 2) -> None:
        super().__init__(modifiable_task_attributes={"target_areas":all_areas}, max_update = max_update)    
    
    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        
        n = random.randint(1,self.max_update)

        all_modif = []
        self.att_state = deepcopy(state.attributes)

        for _ in range(n):
            out = self._get_random_key_value(["target_areas"], self.att_state)
            if out is None:
                break
            self.att_state[out[0]].append(out[1])
            all_modif.append(out[1])

        stages = []
        if all_modif:
            instruction = f"Please add {' and '.join(all_modif)} to your known areas"
            ins = UserInstruction(instruction)

            for i, area in enumerate(all_modif):
                stages.append(
                    ModifAttributesBaseStage(
                        mode = "ADD",
                        instruction=ins,
                        val_name=area,
                        att_name="target_areas",
                        memory=state.memory,
                        preserved_memory_indices=state.preserved_memory_indices,
                        attributes=state.attributes,
                        flag_answer_to_user= i == len(all_modif)-1
                    )
                )
                ins = EmptyInstruction()
        
        return stages
    
class RemoveAreas(RemoveValueToListRequest):

    def __init__(self, max_update : int = 2) -> None:
        super().__init__(["target_areas"], max_update = max_update)
    
    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        
        n = random.randint(1,self.max_update)

        all_modif = []
        self.att_state = deepcopy(state.attributes)

        for _ in range(n):
            out = self._get_random_key_value(["target_areas"], self.att_state)
            if out is None:
                break
            self.att_state[out[0]].remove(out[1])
            all_modif.append(out[1])

        stages = []
        if all_modif:
            instruction = f"Please remove {' and '.join(all_modif)} from your knowledge base."
            ins = UserInstruction(instruction)

            for i, area in enumerate(all_modif):
                stages.append(
                    ModifAttributesBaseStage(
                        mode = "REMOVE",
                        instruction=ins,
                        val_name=area,
                        att_name="target_areas",
                        memory=state.memory,
                        preserved_memory_indices=state.preserved_memory_indices,
                        attributes=state.attributes,
                        flag_answer_to_user= i == len(all_modif)-1
                    )
                )
                ins = EmptyInstruction()
        
        return stages
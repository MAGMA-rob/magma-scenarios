from typing import Dict, Any, List
import random
from copy import deepcopy

from magma_core.base.stage import BaseTaskStage, ModifAttributesBaseStage
from magma_core.base.user_request import BaseRequest
from magma_core.base.state import TaskState
from magma_core.base.data_structures import UserInstruction, EmptyInstruction

from .stages import ObjectToZone
from magma_scenarios.templates.stages import MissingInformationStage, ForbiddenElemStage, Cycle
from magma_scenarios.templates.constraints import RelationAssignmentConstraint
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

        def append_stage(stage: BaseTaskStage) -> None:
            if len(stages) > 0:
                stage.linked_to_prev = True
            stages.append(stage)

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
        has_c = False
        if base_assignement:
            instruction_str += " And consider "
            for obj, area in base_assignement.items():
                instruction_str += f"{obj} to {area}"
            instruction_str += " as news default assignment."
            has_c = True
        cycle_instruction = UserInstruction(instruction_str, has_constraint=has_c)

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
            append_stage(s)

            # TO DO: Random override (to keep rules respect some times)
            cycle_instruction = UserInstruction("Please override these orders just for my cycle")

        if missing_assignment:
            objs = " and ".join(missing_assignment.keys())
            append_stage(MissingInformationStage(
                instruction=cycle_instruction,
                answer=f"The robot must ask about target areas for {objs}",
                memory=[],
                attributes=state.attributes
            ))
            ins = "For this cycle, "
            for o, a in missing_assignment.items():
                ins += f"{o} goes to {a}, "
            cycle_instruction = UserInstruction(ins)
        
        append_stage(Cycle(
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

    constraints : list[RelationAssignmentConstraint]

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
            self.constraints.append(
                RelationAssignmentConstraint(
                    source_value=all_objects[i],
                    target_value=a,
                    relation_key="object_area",
                    source_attribute_key="objects",
                    target_attribute_key="target_areas",
                )
            )
        
        
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
    
class CycleByCategoriesRequest(BaseRequest):

    def __init__(self, max_categories_per_cycle_request : int = 2, max_added_objects_per_empty_category : int = 1) -> None:
        super().__init__()
        self.max_categories = max_categories_per_cycle_request
        self.max_added_objects = max_added_objects_per_empty_category

    def _get_known_types(self, state: TaskState) -> List[str]:
        known_types = list(state.relations.get("type_area", {}).keys())
        for obj_type in state.relations.get("object_type", {}).values():
            if obj_type not in known_types:
                known_types.append(obj_type)
        return known_types

    def _build_cycle_request_instruction(self, categories: List[str]) -> UserInstruction:
        if len(categories) == 1:
            return UserInstruction(f"Launch a cycle for all objects from category {categories[0]}.")
        return UserInstruction(
            f"Launch a cycle for all objects from categories {' and '.join(categories)}."
        )

    def _build_category_resolution_instruction(
            self,
            forgotten_types : List[str],
            added_objects_by_type : Dict[str, List[str]]
        ) -> UserInstruction:
        parts = []
        for obj_type in forgotten_types:
            parts.append(f"Ok, forget category {obj_type} for this cycle")
        for obj_type, objs in added_objects_by_type.items():
            obj_str = " and ".join(objs)
            parts.append(f"For this cycle, consider {obj_str} as {obj_type}")
        return UserInstruction(". ".join(parts) + ".")

    def _build_missing_assignment_instruction(self, missing_type_assignment : Dict[str, str]) -> UserInstruction:
        parts = []
        for obj_type, area in missing_type_assignment.items():
            parts.append(f"For this cycle, category {obj_type} goes to {area}")
        return UserInstruction(". ".join(parts) + ".")

    def _build_area_override_instruction(self) -> UserInstruction:
        return UserInstruction("Please override these area constraints just for this cycle.")

    def sampling_weight(self, state: TaskState) -> float:
        
        if len(state.properties.get("forbidden_objects", [])) > 0 or len(state.properties.get("forbidden_areas", [])) > 0:
            return 0 # not yet implemented

        if len(self._get_known_types(state)) <= 0:
            return 0
        if len(state.attributes.get("objects", [])) <= 0:
            return 0
        if len(state.attributes.get("target_areas", [])) <= 0:
            return 0
        return 1

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        all_types : List = self._get_known_types(state)
        all_objects : List = state.attributes.get("objects", []).copy()
        all_areas : List = state.attributes.get("target_areas", [])

        if len(all_types) <= 0 or len(all_objects) <= 0 or len(all_areas) <= 0:
            raise RuntimeError(f"Failed to build the stage from {self.__class__.__name__} due to empty categories, objects or areas")

        nb_cat = random.randint(1, min(self.max_categories, len(all_types)))
        random.shuffle(all_types)
        selected_types = all_types[:nb_cat]

        object_types = state.relations.get("object_type", {})
        type_areas = state.relations.get("type_area", {})

        type_to_objects = {}
        available_objects = [o for o in all_objects if not o in list(object_types.keys())]
        empty_types = []

        for obj_type in selected_types:
            # build tous les objets associé à ce type 
            associated_objects = [
                obj_name
                for obj_name, associated_type in object_types.items()
                if associated_type == obj_type
            ]

            if associated_objects:
                type_to_objects[obj_type] = associated_objects
            else:
                empty_types.append(obj_type)

        forgotten_types = [] #type to not use
        replaced_categories = [] #type to replace forgotten one
        added_objects_by_type = {} #answer of the human to add object to a type

        if empty_types: # si on a sampler des types sans objets
            cycle_already_possible = any(type_to_objects.values())
            if cycle_already_possible: #si on a des cycles qui fonctionnent on oublie juste
                forgotten_types.extend(empty_types)
            else:
                for t in empty_types:
                    if len(available_objects) > 0:
                        # on ajoute à une nouvelle categorie
                        new_objects = random.choices(available_objects,k=random.randint(1,min(self.max_added_objects, len(available_objects))))
                        added_objects_by_type[t] = new_objects
                        #clean available
                        for o in new_objects:
                            available_objects.remove(o)
                    else: 
                        # pas d'objet sans type on oublie juste pour le moment
                        # TODO: Choisir une categorie au hasard pour remplacer
                        forgotten_types.append(t)
                    

        if not any(type_to_objects.values()):
            raise RuntimeError(
                f"The {self.__class__.__name__} failed to build a cycle because no object could be associated to the selected categories"
            )

        stages = []

        def append_stage(stage: BaseTaskStage) -> None:
            if len(stages) > 0:
                stage.linked_to_prev = True
            stages.append(stage)
        current_instruction = self._build_cycle_request_instruction(selected_types)

        if empty_types:
            empty_types_str = " and ".join(empty_types)
            verb = "has" if len(empty_types) == 1 else "have"
            append_stage(
                ForbiddenElemStage(
                    instruction=current_instruction,
                    answer=f"The model must inform that {empty_types_str} {verb} no associated object.",
                    memory=[],
                    attributes=state.attributes
                )
            )
            current_instruction = self._build_category_resolution_instruction(
                forgotten_types=forgotten_types,
                added_objects_by_type=added_objects_by_type
            )

        # Maintenant on build les assignments par type
        missing_type_assignment = {}
        assignment = {}
        for obj_type, objs in type_to_objects.items():
            target_area = type_areas.get(obj_type, "none")
            if target_area not in all_areas:
                target_area = random.choice(all_areas)
                missing_type_assignment[obj_type] = target_area

            for obj_name in objs:
                assignment[obj_name] = target_area

        if missing_type_assignment:
            missing_types_str = " and ".join(missing_type_assignment.keys())
            append_stage(
                MissingInformationStage(
                    instruction=current_instruction,
                    answer=f"The robot must ask about target areas for {missing_types_str}",
                    memory=[],
                    attributes=state.attributes,
                )
            )
            current_instruction = self._build_missing_assignment_instruction(missing_type_assignment)
        
        # on verifie les interdictions
        # forbidden_object = []
        # obj_with_forbidden_areas = []
        # for obj_name, target_area in assignment.items():
        #     if obj_name in state.properties.get("forbidden_objects", []):
        #         forbidden_object.append(obj_name)
        #     if target_area in state.properties.get("forbidden_areas", []):
        #         obj_with_forbidden_areas.append(obj_name)

        # if obj_with_forbidden_areas or forbidden_object:
        #     objs = " and ".join(forbidden_object)
        #     areas = " and ".join([assignment[o] for o in obj_with_forbidden_areas])

        #     answer = "The model must inform that "
        #     if forbidden_object:
        #         answer += f"{objs} are forbidden"
        #     if obj_with_forbidden_areas:
        #         answer += f"{areas} can not be used."

        #     stages.append(
        #         ForbiddenElemStage(
        #             instruction=current_instruction,
        #             answer=answer,
        #             memory=[],
        #             attributes=state.attributes
        #         )
        #     )
        #     current_instruction = self._build_area_override_instruction()

        append_stage(Cycle(
            assignment=assignment,
            known_areas=all_areas,
            flag_answer=True,
            instruction=current_instruction
        ))

        return stages

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

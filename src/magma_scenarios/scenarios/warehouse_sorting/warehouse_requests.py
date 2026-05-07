from typing import Dict, Any, List
import random
from copy import deepcopy

from magma_core.base.stage import BaseTaskStage, ModifAttributesBaseStage
from magma_core.base.user_request import BaseRequest, BaseConstraintRequest
from magma_core.base.state import TaskState
from magma_core.base.constraints import BaseConstraint
from magma_core.base.data_structures import UserInstruction, EmptyInstruction

from .stages import ObjectToZone
from magma_scenarios.templates.stages import MissingInformationStage, ForbiddenElemStage, Cycle
from magma_scenarios.templates.constraints import RelationAssignmentConstraint
from magma_scenarios.templates.requests import AddValueToListRequest, RemoveValueToListRequest


def _join_values(values: List[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _is_or_are(values: List[str]) -> str:
    return "is" if len(values) == 1 else "are"


class ForbidObjectConstraint(BaseConstraint):
    """Persistently mark one object as forbidden for future cycles."""

    def __init__(self, object_name: str) -> None:
        super().__init__()
        self.object_name = object_name

    def apply(self, state: TaskState):
        super().apply(state)
        if self.object_name not in state.attributes.get("objects", []):
            raise RuntimeError(f"The {self.__class__.__name__} failed to be applied")
        state.properties["forbidden_objects"] = [self.object_name]

    def outdated(self, state: TaskState) -> bool:
        return self.object_name not in state.attributes.get("objects", [])


class AllowObjectConstraint(BaseConstraint):
    """Remove the current object interdiction."""

    def __init__(self, object_name: str) -> None:
        super().__init__()
        self.object_name = object_name

    def apply(self, state: TaskState):
        super().apply(state)
        state.properties["forbidden_objects"] = []

    def outdated(self, state: TaskState) -> bool:
        return self.object_name not in state.attributes.get("objects", [])


class ForbidObjectsRequest(BaseConstraintRequest):
    """Toggle a single persistent object interdiction."""

    def __init__(self) -> None:
        super().__init__()

    def sampling_weight(self, state: TaskState) -> float:
        all_objects = state.attributes.get("objects", [])
        if len(all_objects) <= 0:
            return 0
        if state.properties.get("forbidden_objects"):
            return 0.75
        if len(all_objects) <= 1:
            return 0
        return 1

    def initialize_constraints(self, state: TaskState):
        self.constraints = []
        forbidden_objects = [
            obj
            for obj in state.properties.get("forbidden_objects", [])
            if obj in state.attributes.get("objects", [])
        ]
        if forbidden_objects:
            object_name = forbidden_objects[0]
            self.constraints = [AllowObjectConstraint(object_name)]
            templates = (
                f"{object_name} can be used again now.",
                f"From now on, {object_name} is available again for sorting cycles.",
                f"The maintenance is done: you can manipulate {object_name} again.",
            )
            self.constraint_msg = random.choice(templates)
            return

        available_objects = [
            obj
            for obj in state.attributes.get("objects", [])
        ]
        if len(available_objects) <= 1:
            raise RuntimeError(
                f"Failed to build the stage from {self.__class__.__name__} "
                "due to too few available objects"
            )

        object_name = random.choice(available_objects)
        self.constraints = [ForbidObjectConstraint(object_name)]

        templates = (
            f"Please remember that {object_name} must not be manipulated for now.",
            f"New safety rule: do not sort {object_name} until I say otherwise.",
            f"From now on, {object_name} cannot be used in sorting cycles.",
        )
        self.constraint_msg = random.choice(templates)


class TemporaryObjectAssignmentCycleRequest(BaseRequest):
    """Launch a cycle with a one-shot assignment that does not update defaults."""

    def __init__(
            self,
            max_object_per_cycle_request: int = 3,
            all_objects_probability: float = 0.7,
        ) -> None:
        super().__init__()
        self.max_object = max_object_per_cycle_request
        self.all_objects_probability = all_objects_probability

    def sampling_weight(self, state: TaskState) -> float:
        if len(state.attributes.get("objects", [])) <= 0:
            return 0
        if len(state.attributes.get("target_areas", [])) <= 0:
            return 0
        if state.properties.get("object_area_needs_application", False):
            return 0.5
        return 2

    def _sample_objects(self, objects: List[str]) -> List[str]:
        if random.random() < self.all_objects_probability:
            return objects.copy()
        shuffled_objects = objects.copy()
        random.shuffle(shuffled_objects)
        nb_objects = random.randint(1, min(self.max_object, len(shuffled_objects)))
        return shuffled_objects[:nb_objects]

    def _build_instruction(
            self,
            objects_to_sort: List[str],
            target_area: str,
            all_objects: List[str],
        ) -> UserInstruction:
        if len(objects_to_sort) == len(all_objects):
            templates = (
                f"Launch a cycle sending all objects to {target_area}.",
                f"For this cycle, send every object to {target_area}.",
                f"Right now, all objects should go to {target_area}.",
            )
        else:
            obj_text = _join_values(objects_to_sort)
            templates = (
                f"Launch a cycle sending {obj_text} to {target_area}.",
                f"For this cycle, send {obj_text} to {target_area}.",
                f"Right now, {obj_text} should go to {target_area}.",
            )
        return UserInstruction(random.choice(templates), has_constraint=True)

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        all_objects = state.attributes.get("objects", []).copy()
        all_areas = state.attributes.get("target_areas", [])
        if len(all_objects) <= 0 or len(all_areas) <= 0:
            raise RuntimeError(
                f"Failed to build the stage from {self.__class__.__name__} "
                "due to empty objects or areas"
            )

        objects_to_sort = self._sample_objects(all_objects)
        target_area = random.choice(all_areas)
        instruction = self._build_instruction(objects_to_sort, target_area, all_objects)

        forbidden_objects = [
            obj
            for obj in objects_to_sort
            if obj in state.properties.get("forbidden_objects", [])
        ]
        if forbidden_objects:
            return [
                ForbiddenElemStage(
                    instruction=instruction,
                    answer=(
                        f"The model must inform that {_join_values(forbidden_objects)} "
                        f"{_is_or_are(forbidden_objects)} forbidden"
                    ),
                    memory=[],
                    attributes=state.attributes,
                )
            ]

        if target_area in state.properties.get("forbidden_areas", []):
            return [
                ForbiddenElemStage(
                    instruction=instruction,
                    answer=f"The model must inform that {target_area} can not be used.",
                    memory=[],
                    attributes=state.attributes,
                )
            ]

        return [
            Cycle(
                assignment={obj: target_area for obj in objects_to_sort},
                known_areas=all_areas,
                flag_answer=True,
                instruction=instruction,
            )
        ]


class CycleRequest(BaseRequest):

    def __init__(self, max_object_per_cycle_request : int = 3) -> None:
        super().__init__()
        self.max_object = max_object_per_cycle_request

    def sampling_weight(self, state: TaskState) -> float:
        if len(state.attributes.get("objects", [])) <= 0:
            return 0
        if len(state.attributes.get("target_areas", [])) <= 0:
            return 0
        if state.properties.get("object_area_needs_application", False):
            return 8
        object_assignments = state.relations.get("object_area", {})
        if len(object_assignments) > 0:
            return 4
        return 2

    def _sample_objects_to_sort(self, state: TaskState) -> List[str]:
        all_objects = state.attributes.get("objects", []).copy()
        random.shuffle(all_objects)
        max_objects = min(self.max_object, len(all_objects))
        nb_obj = random.randint(1, max_objects)

        object_assignments = state.relations.get("object_area", {})
        assigned_objects = [
            obj for obj in all_objects
            if obj in object_assignments
        ]
        missing_objects = [
            obj for obj in all_objects
            if obj not in object_assignments
        ]

        selected: List[str] = []
        if state.properties.get("object_area_needs_application", False) and assigned_objects:
            selected.append(random.choice(assigned_objects))
        elif object_assignments and missing_objects and random.random() < 0.5:
            selected.append(random.choice(missing_objects))
            if len(selected) < nb_obj and assigned_objects:
                selected.append(random.choice(assigned_objects))

        for obj in all_objects:
            if len(selected) >= nb_obj:
                break
            if obj not in selected:
                selected.append(obj)

        return selected

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
            rule_text = ", ".join(
                f"{obj} to {area}"
                for obj, area in base_assignement.items()
            )
            assignment_word = "assignments" if len(base_assignement) > 1 else "assignment"
            instruction_str += f" And consider {rule_text} as new default {assignment_word}."
            has_c = True
        cycle_instruction = UserInstruction(instruction_str, has_constraint=has_c)

        if obj_with_forbidden_areas or forbidden_object:
            objs = " and ".join(forbidden_object)
            areas = " and ".join([assignement[o] for o in obj_with_forbidden_areas])

            answer = "The model must inform that "
            if forbidden_object:
                answer+= f"{objs} {_is_or_are(forbidden_object)} forbidden"
            if obj_with_forbidden_areas:
                answer+= f"{areas} can not be used."

            s = ForbiddenElemStage(
                instruction=cycle_instruction,
                answer=answer,
                memory=[],
                attributes=state.attributes
            )
            append_stage(s)
            return stages

        if missing_assignment:
            objs = " and ".join(missing_assignment.keys())
            append_stage(MissingInformationStage(
                instruction=cycle_instruction,
                answer=f"The robot must ask about target areas for {objs}",
                memory=[],
                attributes=state.attributes
            ))
            assignment_text = ", ".join(
                f"{obj} goes to {area}"
                for obj, area in missing_assignment.items()
            )
            cycle_instruction = UserInstruction(f"For this cycle, {assignment_text}.")
        
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
        
        return self._create_stages(self._sample_objects_to_sort(state), all_areas, state)

    def apply_request(self, state: TaskState) -> TaskState:
        if state.properties.get("object_area_needs_application", False):
            state.properties["object_area_needs_application"] = False
            state.properties["object_area_applications"] = (
                state.properties.get("object_area_applications", 0) + 1
            )
        return state

class CycleWithPermanentRulesRequest(CycleRequest):

    constraints : list[RelationAssignmentConstraint]

    def __init__(self, max_object_per_cycle_request: int = 3, max_permanent_rule : int = 2) -> None:
        super().__init__(max_object_per_cycle_request)
        self.max_rules = max_permanent_rule

    def sampling_weight(self, state: TaskState) -> float:
        if state.properties.get("object_area_needs_application", False):
            return 1
        if len(state.relations.get("object_area", {})) < 1:
            return 3
        return 1

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
        if len(self.constraints) > 0:
            state.properties["object_area_needs_application"] = True
        return state
        

class MoveOneObjectRequest(BaseRequest):

    def __init__(self) -> None:
        super().__init__()

    def sampling_weight(self, state: TaskState) -> float:
        if state.properties.get("object_area_needs_application", False):
            return 0.5
        return 1

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

    def __init__(self, max_categories_per_cycle_request : int = 2) -> None:
        super().__init__()
        self.max_categories = max_categories_per_cycle_request

    def _get_known_types(self, state: TaskState) -> List[str]:
        known_objects = set(state.attributes.get("objects", []))
        known_types = []
        for obj_name, obj_type in state.relations.get("object_type", {}).items():
            if obj_name not in known_objects:
                continue
            if obj_type not in known_types:
                known_types.append(obj_type)
        return known_types

    def _build_cycle_request_instruction(self, categories: List[str]) -> UserInstruction:
        if len(categories) == 1:
            return UserInstruction(f"Launch a cycle for all objects from category {categories[0]}.")
        return UserInstruction(
            f"Launch a cycle for all objects from categories {' and '.join(categories)}."
        )

    def _build_missing_assignment_instruction(self, missing_type_assignment : Dict[str, str]) -> UserInstruction:
        parts = []
        for obj_type, area in missing_type_assignment.items():
            parts.append(f"For this cycle, category {obj_type} goes to {area}")
        return UserInstruction(". ".join(parts) + ".")

    def sampling_weight(self, state: TaskState) -> float:
        
        if len(state.properties.get("forbidden_objects", [])) > 0 or len(state.properties.get("forbidden_areas", [])) > 0:
            return 0 # not yet implemented

        if len(self._get_known_types(state)) <= 0:
            return 0
        if len(state.attributes.get("objects", [])) <= 0:
            return 0
        if len(state.attributes.get("target_areas", [])) <= 0:
            return 0
        if state.properties.get("type_area_needs_application", False):
            return 8
        if state.properties.get("object_type_needs_application", False):
            return 5
        if len(state.relations.get("type_area", {})) > 0:
            return 4
        return 2

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

        for obj_type in selected_types:
            associated_objects = [
                obj_name
                for obj_name, associated_type in object_types.items()
                if associated_type == obj_type and obj_name in all_objects
            ]
            type_to_objects[obj_type] = associated_objects

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

        append_stage(Cycle(
            assignment=assignment,
            known_areas=all_areas,
            flag_answer=True,
            instruction=current_instruction
        ))

        return stages

    def apply_request(self, state: TaskState) -> TaskState:
        if state.properties.get("object_type_needs_application", False):
            state.properties["object_type_needs_application"] = False
        if state.properties.get("type_area_needs_application", False):
            state.properties["type_area_needs_application"] = False
        state.properties["category_cycle_applications"] = (
            state.properties.get("category_cycle_applications", 0) + 1
        )
        return state

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

    def sampling_weight(self, state: TaskState) -> float:
        target_areas = state.attributes.get("target_areas", [])
        if len(target_areas) <= 1:
            return 0
        return super().sampling_weight(state)
    
    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        
        n = random.randint(1,self.max_update)

        all_modif = []
        self.att_state = deepcopy(state.attributes)

        for _ in range(n):
            if len(self.att_state["target_areas"]) == 1:
                break
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

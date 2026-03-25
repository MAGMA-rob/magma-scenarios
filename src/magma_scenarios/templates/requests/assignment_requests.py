from typing import Type, Dict, Any, List
import random
from collections import defaultdict

from magma_core.base.state.task_state import TaskState
from magma_core.base.user_request import BaseConstraintRequest

from ..constraints import RelationAssignmentConstraint

class GiveObjectAssignmentRequest(BaseConstraintRequest):
    """Sample direct object-to-area rules and expose them as one constraint request."""

    def __init__(self, max_simultaneous_change : int = 1):
        super().__init__()
        self.max_change = max_simultaneous_change

    def sampling_weight(self, state: TaskState) -> float:
        if len(state.relations.get("object_area",{})) < 1:
            return 3 # if no assignment, probability to sample this request increase.
        return 1
    
    def initialize_constraints(self, state: TaskState):
        self.constraints = []
        all_objects = state.attributes.get("objects", []).copy()
        all_areas = state.attributes.get("target_areas", [])

        if len(all_objects) <= 0 or len(all_areas) <=0:
            raise RuntimeError(f"Failed to build the stage from {self.__class__.__name__} due to empty objects or areas")
        
        max_val = min(self.max_change,len(all_objects))
        nb_change = random.randint(1,max_val)

        random.shuffle(all_objects)
        selected_areas = random.choices(all_areas, k=nb_change)

        self.constraint_msg = "Hey, here are some sorting rules: "
        for i in range(nb_change):
            self.constraints.append(
                RelationAssignmentConstraint(
                    source_value=all_objects[i],
                    target_value=selected_areas[i],
                    relation_key="object_area",
                    source_attribute_key="objects",
                    target_attribute_key="target_areas",
                )
            )
            self.constraint_msg += f"{all_objects[i]} goes to {selected_areas[i]}"
            if i < nb_change -1:
                self.constraint_msg += ","
        self.constraint_msg += "."

class GiveObjectCategoryRequest(BaseConstraintRequest):
    """Sample object-to-category updates for sorting tasks."""

    categories : List[str]

    def __init__(self, available_categories : List[str], max_object_assignment : int = 2):
        super().__init__()
        if len(available_categories) < 2:
            raise ValueError("It must have at least 2 caegories")
        self.categories = available_categories
        self.max_obj = max_object_assignment

    def sampling_weight(self, state: TaskState) -> float:
        if len(state.relations.get("object_type",{})) < 1:
            return 3 # if no assignment, probability to sample this request increase.
        return 1

    def initialize_constraints(self, state: TaskState):
        self.constraints = []
        all_objects = state.attributes.get("objects", []).copy()

        if len(all_objects) <=0:
            raise RuntimeError(f"Failed to build the stage from {self.__class__.__name__} due to empty objects")
        
        random.shuffle(all_objects)
        nb_update = random.randint(1,min(self.max_obj, len(all_objects)))

        assignment = defaultdict(list)
        for i in range(nb_update):
            cur_t = state.relations["object_type"].get(all_objects[i],None)
            t = random.choice(self.categories)
            if cur_t != t:
                assignment[t].append(all_objects[i])
                self.constraints.append(
                    RelationAssignmentConstraint(
                        source_value=all_objects[i],
                        target_value=t,
                        relation_key="object_type",
                        source_attribute_key="objects",
                    )
                )
        
        self.constraint_msg = "Hello,"
        for t, objs in assignment.items():
            obj_str = " and ".join(objs)
            self.constraint_msg += f" {obj_str} are now {t},"
        self.constraint_msg += "."

class GiveCategoryAssignmentRequest(BaseConstraintRequest):
    """Sample category-to-area routing rules for sorting tasks."""

    categories : List[str]

    def __init__(self, available_categories : List[str], max_categories_assignment : int = 2):
        super().__init__()
        if len(available_categories) < 2:
            raise ValueError("It must have at least 2 caegories")
        self.categories = available_categories
        self.max_categories = max_categories_assignment

    def sampling_weight(self, state: TaskState) -> float:
        if len(state.relations.get("type_area",{})) < 1:
            return 3 # if no assignment, probability to sample this request increase.
        return 1

    def initialize_constraints(self, state: TaskState):
        self.constraints = []
        all_area = state.attributes.get("target_areas", [])

        if len(all_area) <=0:
            raise RuntimeError(f"Failed to build the stage from {self.__class__.__name__} due to empty objects")

        known_type = [v for _, v in state.relations["object_type"].items()]
        nb_update = random.randint(1,min(self.max_categories, len(self.categories)))
        # We choose in priority existant categories and we add random ones to reach the desired number.
        if len(known_type) < nb_update:
            random.shuffle(self.categories)
            categories = known_type + self.categories.copy()
        else:
            categories = known_type

        assignment = defaultdict(list)
        for i in range(nb_update):
            cur_area = state.relations["type_area"].get(categories[i],None)
            target_area = random.choice(all_area)
            if cur_area != target_area:
                assignment[target_area].append(categories[i])
                self.constraints.append(
                    RelationAssignmentConstraint(
                        source_value=categories[i],
                        target_value=target_area,
                        relation_key="type_area",
                        target_attribute_key="target_areas",
                    )
                )
        
        self.constraint_msg = ""
        for area, types in assignment.items():
            types_str = " and ".join(types)
            self.constraint_msg += f" {types_str} are now going to {area},"
        self.constraint_msg += "."

from typing import Callable, Dict, List, Optional, Tuple
import random
from collections import defaultdict

from magma_core.base.state.task_state import TaskState
from magma_core.base.user_request import BaseConstraintRequest
from magma_core.base.constraints import BaseConstraint

from ..constraints import RelationAssignmentConstraint

ConstraintBuilder = Callable[[str, str], BaseConstraint]
AssignmentMessageBuilder = Callable[[str, List[Tuple[str, str]]], str]


class GiveRelationAssignmentRequest(BaseConstraintRequest):
    """Sample direct source-to-target relation rules from task-state attributes."""

    def __init__(
            self,
            relation_key: str,
            source_attribute_key: str,
            target_attribute_key: str,
            max_simultaneous_change: int = 1,
            empty_relation_sampling_weight: float = 3,
            intro_message: str = "Hey, here are some rules: ",
            assignment_template: str = "{source} goes to {target}",
            constraint_builder: Optional[ConstraintBuilder] = None,
            constraint_message_builder: Optional[AssignmentMessageBuilder] = None,
        ) -> None:
        super().__init__()
        self.relation_key = relation_key
        self.source_attribute_key = source_attribute_key
        self.target_attribute_key = target_attribute_key
        self.max_change = max_simultaneous_change
        self.empty_relation_sampling_weight = empty_relation_sampling_weight
        self.intro_message = intro_message
        self.assignment_template = assignment_template
        self.constraint_builder = constraint_builder
        self.constraint_message_builder = constraint_message_builder

    def _build_constraint(self, source_value: str, target_value: str) -> BaseConstraint:
        if self.constraint_builder is not None:
            return self.constraint_builder(source_value, target_value)
        return RelationAssignmentConstraint(
            source_value=source_value,
            target_value=target_value,
            relation_key=self.relation_key,
            source_attribute_key=self.source_attribute_key,
            target_attribute_key=self.target_attribute_key,
        )

    def sampling_weight(self, state: TaskState) -> float:
        if len(state.relations.get(self.relation_key, {})) < 1:
            return self.empty_relation_sampling_weight
        return 1

    def initialize_constraints(self, state: TaskState):
        self.constraints = []
        all_sources = state.attributes.get(self.source_attribute_key, []).copy()
        all_targets = state.attributes.get(self.target_attribute_key, [])

        if len(all_sources) <= 0 or len(all_targets) <= 0:
            raise RuntimeError(
                f"Failed to build the stage from {self.__class__.__name__} "
                f"due to empty {self.source_attribute_key} or {self.target_attribute_key}"
            )

        max_val = min(self.max_change, len(all_sources))
        nb_change = random.randint(1, max_val)

        random.shuffle(all_sources)
        selected_targets = random.choices(all_targets, k=nb_change)
        assignments: List[Tuple[str, str]] = []
        for i in range(nb_change):
            source_value = all_sources[i]
            target_value = selected_targets[i]
            assignments.append((source_value, target_value))
            self.constraints.append(
                self._build_constraint(source_value, target_value)
            )

        if self.constraint_message_builder is not None:
            self.constraint_msg = self.constraint_message_builder(self.intro_message, assignments)
            return

        self.constraint_msg = self.intro_message
        for i, (source_value, target_value) in enumerate(assignments):
            self.constraint_msg += self.assignment_template.format(
                source=source_value,
                target=target_value,
            )
            if i < len(assignments) - 1:
                self.constraint_msg += ", "
        self.constraint_msg += "."


class GiveObjectAssignmentRequest(GiveRelationAssignmentRequest):
    """Sample direct object-to-area rules and expose them as one constraint request."""

    def __init__(self, max_simultaneous_change : int = 1):
        super().__init__(
            relation_key="object_area",
            source_attribute_key="objects",
            target_attribute_key="target_areas",
            max_simultaneous_change=max_simultaneous_change,
            intro_message="Hey, here are some sorting rules: ",
            assignment_template="{source} goes to {target}",
        )

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
        if len(state.relations.get("type_area",{})) > 3:
            return 0 # AVoiding too much category
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

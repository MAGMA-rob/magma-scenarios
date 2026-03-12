from typing import Type, Dict, Any, List
import random

from magma_core.base.state.task_state import TaskState
from magma_core.base.user_request import BaseConstraintRequest

from ..constraints import ObjectAssignmentConstraint

class GiveObjectAssignmentRequest(BaseConstraintRequest):

    def __init__(self, max_simultaneous_change : int = 1):
        super().__init__()
        self.max_change = max_simultaneous_change
    
    def initialize_constraints(self, state: TaskState):
        self.constraints = []
        all_objects = state.entities.get("objects", []).copy()
        all_areas = state.entities.get("zones", [])

        if len(all_objects) <= 0 or len(all_areas) <=0:
            raise RuntimeError(f"Failed to build the stage from {self.__class__.__name__} due to empty objects or areas")
        
        max_val = min(self.max_change,len(all_objects))
        nb_change = random.randint(1,max_val)

        random.shuffle(all_objects)
        selected_areas = random.choices(all_areas, k=nb_change)

        for i in range(nb_change):
            self.constraints.append(ObjectAssignmentConstraint(all_objects[i],selected_areas[i]))
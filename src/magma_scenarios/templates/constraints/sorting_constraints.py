from magma_core.base.constraints import BaseConstraint
from magma_core.base.state.task_state import TaskState

class ObjectAssignmentConstraint(BaseConstraint):
    """Assign one object to one target area in the latent sorting state."""

    def __init__(self, object : str, zone : str) -> None:
        super().__init__()
        self.obj = object
        self.zone = zone

    def apply(self, state: TaskState):
        super().apply(state) # important for register the constraint in the state list.
        all_obj = state.attributes.get("objects", [])
        all_zone = state.attributes.get("target_areas", [])
        if not self.obj in all_obj or not self.zone in all_zone:
            raise RuntimeError(f"The {self.__class__.__name__} failed to be applied")
        state.relations["object_area"][self.obj] = self.zone

    def outdated(self, state: TaskState) -> bool:
        if (not self.obj in state.attributes.get("objects",[]) 
            or not self.zone in state.attributes.get("target_areas",[])):
            return True
        return False
    
class ObjectCategoryConstraint(BaseConstraint):
    """Assign one object to a category in the latent sorting state."""

    def __init__(self, object : str, category : str) -> None:
        super().__init__()
        self.obj = object
        self.category = category

    def apply(self, state: TaskState):
        super().apply(state) # important for register the constraint in the state list.
        all_obj = state.attributes.get("objects", [])
        if not "object_type" in state.relations:
            state.relations["object_type"] = {}
        if not self.obj in all_obj:
            raise RuntimeError(f"The {self.__class__.__name__} failed to be applied")
        state.relations["object_type"][self.obj] = self.category

    def outdated(self, state: TaskState) -> bool:
        if not self.obj in state.attributes.get("objects",[]):
            return True
        return False
    
class CategoryAreaConstraint(BaseConstraint):
    """Assign one category to one target area in the latent sorting state."""

    def __init__(self, category : str, zone : str) -> None:
        super().__init__()
        self.zone = zone
        self.category = category

    def apply(self, state: TaskState):
        super().apply(state) # important for register the constraint in the state list.
        all_zone = state.attributes.get("target_areas", [])
        if not "type_area" in state.relations:
            state.relations["type_area"] = {}
        if not self.zone in all_zone:
            raise RuntimeError(f"The {self.__class__.__name__} failed to be applied")
        state.relations["type_area"][self.category] = self.zone

    def outdated(self, state: TaskState) -> bool:
        if not self.zone in state.attributes.get("areas",[]):
            return True
        return False

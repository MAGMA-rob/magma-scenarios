from magma_core.base.constraints import BaseConstraint
from magma_core.base.state.task_state import TaskState

class ObjectAssignmentConstraint(BaseConstraint):

    def __init__(self, object : str, zone : str) -> None:
        super().__init__()
        self.obj = object
        self.zone = zone

    def apply(self, state: TaskState):
        all_obj = state.entities.get("objects", [])
        all_zone = state.entities.get("zones", [])
        if not self.obj in all_obj or not self.zone in all_zone:
            raise RuntimeError(f"The {self.__class__.__name__} failed to be applied")
        state.relations["object_zone"][self.obj] = self.zone


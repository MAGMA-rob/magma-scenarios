
from magma_core.base.tasks import TaskDefinition
from magma_core.base.state import TaskState
from magma_scenarios.templates.requests import GiveObjectAssignmentRequest

from .requests import MoveOneObjectRequest, CycleRequest, CycleWithPermanentRulesRequest
from .tasks.att import OBJECTS, AREAS

class SimplifiedWarehouseSortingDefinition(TaskDefinition):

    active_requests = [
        MoveOneObjectRequest(),
        GiveObjectAssignmentRequest(max_simultaneous_change=2),
        CycleRequest(),
        CycleWithPermanentRulesRequest()
    ]

    def __init__(self) -> None:
        
        self.starting_state = TaskState()
        self.starting_state.entities["objects"] = OBJECTS
        self.starting_state.entities["zones"] = AREAS
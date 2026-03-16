from magma_core.base.tasks import TaskDefinition
from magma_core.base.state import TaskState
from magma_scenarios.templates.requests import (
    GiveObjectAssignmentRequest,
    GiveCategoryAssignmentRequest,
    GiveObjectCategoryRequest
)

from .requests import MoveOneObjectRequest, CycleRequest, CycleWithPermanentRulesRequest
from .att import OBJECTS, AREAS
from .tools import WithoutManufacturingOrder

from pathlib import Path

class SimpleSortingDefinition(TaskDefinition):

    active_requests = [
        MoveOneObjectRequest(),
        GiveObjectAssignmentRequest(max_simultaneous_change=2),
        CycleRequest(),
        CycleWithPermanentRulesRequest()
    ]
    Tools_cls = WithoutManufacturingOrder
    env_id = "SortingCubesWarehouse-v1"

    def __init__(self) -> None:
        
        self.starting_state = TaskState()
        self.starting_state.entities["objects"] = OBJECTS
        self.starting_state.entities["zones"] = AREAS

        super().__init__(randomized_config_path=str(Path(__file__).parent.joinpath("config.yaml")))

known_category = [
    "Product A", "Product B", "Fragile", "Rare",
    "Category 1", "Category 2", "Category 3", "Dangerous Category",
    "Type X4", "Type J5", "Type 98", "Type 05",
    "Group A", "Group B", "Group C",
    "Waste Group", "Mechanical Group", "Support Pieces"
    ""
]

class SortingCategoryDefinition(TaskDefinition):

    active_requests = [
        MoveOneObjectRequest(),
        GiveObjectAssignmentRequest(max_simultaneous_change=2),
        CycleRequest(),
        GiveObjectCategoryRequest(known_category, max_object_assignment=3),
        GiveCategoryAssignmentRequest(known_category, max_categories_assignment=3)
    ]
    Tools_cls = WithoutManufacturingOrder
    env_id = "SortingCubesWarehouse-v1"

    def __init__(self) -> None:
        
        self.starting_state = TaskState()
        self.starting_state.entities["objects"] = OBJECTS
        self.starting_state.entities["zones"] = AREAS

        super().__init__(randomized_config_path=str(Path(__file__).parent.joinpath("config.yaml")))
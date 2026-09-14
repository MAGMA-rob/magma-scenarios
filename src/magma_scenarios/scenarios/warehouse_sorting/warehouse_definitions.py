from magma_core.simulation.tasks import TaskDefinition
from magma_core.simulation.state import TaskState
from magma_core.simulation.data_structures import SituationInit
from .requests import (
    CycleRequest,
    CycleWithPermanentRulesRequest,
    ForbidObjectsRequest,
    TemporaryObjectAssignmentCycleRequest,
    AddAreas,
    RemoveAreas,
    CycleByCategoriesRequest,
    AskObjectAreaAssignementRequest,
    AskObjectAreaAssignementRequestInverse,
    WarehouseSortingInterruptionRequest,
    WarehouseCategoryAssignmentRequest,
    WarehouseObjectAssignmentRequest,
    WarehouseObjectCategoryRequest,
    )

from .att import OBJECTS, AREAS
from .tools import WithoutManufacturingOrder
from .rule_renderer import WarehouseRuleRenderer

from copy import deepcopy
from pathlib import Path

WAREHOUSE_ATTRIBUTES = {
    "objects": OBJECTS,
    "target_areas": AREAS,
}
WAREHOUSE_RANDOMIZED_CONFIG_PATH = str(Path(__file__).parent.joinpath("config.yaml"))

class SimpleSortingDefinition(TaskDefinition):

    active_requests = [
        AddAreas(AREAS),
        RemoveAreas(),
        WarehouseObjectAssignmentRequest(
            max_simultaneous_change=2,
            existing_relation_sampling_weight=2.5,
            pending_relation_sampling_weight=0.75,
        ),
        CycleRequest(),
        CycleWithPermanentRulesRequest(
            rule_update_weight=2.0,
            pending_rule_update_weight=0.75,
        ),
        AskObjectAreaAssignementRequest(),
        TemporaryObjectAssignmentCycleRequest(),
        WarehouseSortingInterruptionRequest(),
        AskObjectAreaAssignementRequestInverse()
    ]
    Tools_cls = WithoutManufacturingOrder
    maniskill_env_id = "SortingCubesWarehouse-v1"
    randomized_config_path = WAREHOUSE_RANDOMIZED_CONFIG_PATH
    RuleRenderer_cls = WarehouseRuleRenderer

    def __init__(self) -> None:
        attributes = deepcopy(WAREHOUSE_ATTRIBUTES)
        situation_init = SituationInit(
            attributes=attributes,
            all_task_attributes=deepcopy(WAREHOUSE_ATTRIBUTES),
        )
        self.starting_state = TaskState()
        self.starting_state.attributes = deepcopy(WAREHOUSE_ATTRIBUTES)
        super().__init__(situation_init=situation_init)


class SortingWithInterdictionsDefinition(TaskDefinition):
    """Object assignment training with temporary overrides and one forbidden object."""

    active_requests = [
        WarehouseObjectAssignmentRequest(
            max_simultaneous_change=2,
            existing_relation_sampling_weight=2.0,
            pending_relation_sampling_weight=0.75,
        ),
        ForbidObjectsRequest(
            forbid_sampling_weight=1.0,
            allow_sampling_weight=0.75,
            aged_sampling_weight=6.0,
            steps_to_aged_weight=5,
        ),
        CycleWithPermanentRulesRequest(
            rule_update_weight=1.5,
            pending_rule_update_weight=0.75,
        ),
        CycleRequest(forbidden_object_sampling_probability=0.7),
        AskObjectAreaAssignementRequest(),
        AskObjectAreaAssignementRequestInverse()
    ]
    Tools_cls = WithoutManufacturingOrder
    maniskill_env_id = "SortingCubesWarehouse-v1"
    randomized_config_path = WAREHOUSE_RANDOMIZED_CONFIG_PATH
    RuleRenderer_cls = WarehouseRuleRenderer

    def __init__(self) -> None:
        attributes = deepcopy(WAREHOUSE_ATTRIBUTES)
        situation_init = SituationInit(
            attributes=attributes,
            all_task_attributes=deepcopy(WAREHOUSE_ATTRIBUTES),
        )
        self.starting_state = TaskState()
        self.starting_state.attributes = deepcopy(WAREHOUSE_ATTRIBUTES)
        super().__init__(situation_init=situation_init)

known_category = [
    "Product A", "Product B", "Fragile", "Rare",
    "Category 1", "Category 2", "Category 3", "Dangerous Category",
    "Type X4", "Type J5", "Type 98", "Type 05",
    "Group A", "Group B", "Group C",
    "Waste Group", "Mechanical Group", "Support Pieces"
]

class SortingCategoryDefinition(TaskDefinition):

    active_requests = [
        AddAreas(AREAS),
        RemoveAreas(),
        CycleByCategoriesRequest(
            cycle_weight=2.5,
            empty_type_area_weight=1.5,
            pending_object_type_weight=4.0,
            pending_type_area_weight=6.0,
        ),
        WarehouseObjectCategoryRequest(
            known_category,
            max_object_assignment=3,
            existing_relation_sampling_weight=2.0,
            pending_relation_sampling_weight=0.75,
        ),
        WarehouseCategoryAssignmentRequest(
            known_category,
            max_categories_assignment=3,
            existing_relation_sampling_weight=2.5,
            pending_relation_sampling_weight=0.75,
        )
    ]
    Tools_cls = WithoutManufacturingOrder
    maniskill_env_id = "SortingCubesWarehouse-v1"
    randomized_config_path = WAREHOUSE_RANDOMIZED_CONFIG_PATH
    RuleRenderer_cls = WarehouseRuleRenderer

    def __init__(self) -> None:
        attributes = deepcopy(WAREHOUSE_ATTRIBUTES)
        situation_init = SituationInit(
            attributes=attributes,
            all_task_attributes=deepcopy(WAREHOUSE_ATTRIBUTES),
        )
        self.starting_state = TaskState()
        self.starting_state.attributes = deepcopy(WAREHOUSE_ATTRIBUTES)
        super().__init__(situation_init=situation_init)

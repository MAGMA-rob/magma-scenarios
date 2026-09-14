from copy import deepcopy
from pathlib import Path
from magma_core.simulation.data_structures import SituationInit
from magma_core.simulation.state import TaskState
from magma_core.simulation.tasks import InitializationParameters, TaskDefinition
from .brs_attributes import (
    OBJECT_TYPES,
    ROBOTS,
    ZONES,
    build_random_env_start,
    sorting_objects,
)
from .brs_tools import BiRobotTools
from .requests.common import (
    DAMAGED_TYPE_AREA_RELATION_KEY,
    INTACT_TYPE_AREA_RELATION_KEY,
    OBJECT_CONDITION_RELATION_KEY,
    OBJECT_AREAS_KEY,
    condition_assignments_from_env_options,
)
from .rule_renderer import BiRobotSortingRuleRenderer
from .requests import (
    SortAllFruits,
    MoveFruitSequence,
    MoveFruitSet,
    TypePriorityRequest,
    SortDamagedObjectsRequest,
    AskTypeZoneRequest,
    AskZoneContentsRequest,
    AskDamagedObjectsRequest,
    SmallMoveFruitSet,
)


def _sample_environment(
    min_damaged_objects: int = 1,
    max_damaged_objects: int = 3,
) -> dict[str, list[str]]:
    """
    Randomly distribute all fruits between the three zones.

    Damaged objects are disabled for this definition because the first
    version focuses only on sorting, recipes and priorities.
    """

    (
        left_zone,
        mutual_zone,
        right_zone,
        damaged_objects,
    ) = build_random_env_start(
        max_per_zone=4,
        min_damaged_objects=min_damaged_objects,
        max_damaged_objects=max_damaged_objects,
    )

    return {
        "left_zone": left_zone,
        "mutual_zone": mutual_zone,
        "right_zone": right_zone,
        "damaged_objs": damaged_objects,
    }


def _all_object_names() -> list[str]:
    return [
        object_name
        for object_names in sorting_objects.values()
        for object_name in object_names
    ]


class BiRobotSortingDefinition(TaskDefinition):
    name = "Bi Robot Sorting Definition"

    maniskill_env_id = "BiRobotSorting-v1"
    Tools_cls = BiRobotTools
    RuleRenderer_cls = BiRobotSortingRuleRenderer

    randomized_config_path = str(
        Path(__file__).resolve().parent / "config.yaml"
    )

    active_requests = [
        SortAllFruits(),
        MoveFruitSequence(),
        MoveFruitSet(),
        TypePriorityRequest(),
        SortDamagedObjectsRequest(),
        AskTypeZoneRequest(),
        AskZoneContentsRequest(),
        AskDamagedObjectsRequest(),
    ]

    def __init__(
        self,
        min_damaged_objects: int = 1,
        max_damaged_objects: int = 3,
    ) -> None:
        env_options = _sample_environment(
            min_damaged_objects=min_damaged_objects,
            max_damaged_objects=max_damaged_objects,
        )

        attributes = {
            "known_robots": ROBOTS.copy(),
            "locations": ZONES.copy(),
            "object_classes": OBJECT_TYPES.copy(),
        }

        starting_state = TaskState()
        starting_state.attributes = deepcopy(attributes)

        intact_assignment, damaged_assignment = (
            condition_assignments_from_env_options(env_options)
        )

        starting_state.relations[INTACT_TYPE_AREA_RELATION_KEY] = intact_assignment
        starting_state.relations[DAMAGED_TYPE_AREA_RELATION_KEY] = damaged_assignment

        starting_state.relations[OBJECT_CONDITION_RELATION_KEY] = {
            object_name: (
                "damaged"
                if object_name in env_options["damaged_objs"]
                else "intact"
            )
            for object_name in _all_object_names()
        }
        starting_state.properties[OBJECT_AREAS_KEY] = {
            object_name: zone
            for zone in ZONES
            for object_name in env_options[zone]
        }

        initialization_parameters = InitializationParameters(
            env_options=deepcopy(env_options),
            agent_names=ROBOTS.copy(),
        )

        memory = {
            "memory_list": [
                (
                    "arm1 operates in left_zone, while arm2 "
                    "operates in right_zone."
                ),
                (
                    "Objects transferred between left_zone and "
                    "right_zone must pass through mutual_zone."
                ),
                (
                    "Only arm1 can inspect the sorting areas "
                    "with get_objects_state."
                ),
            ]
        }

        super().__init__(
            situation_init=SituationInit(
                attributes=deepcopy(attributes),
                all_task_attributes=deepcopy(attributes),
                memory=memory,
            ),
            starting_state=starting_state,
            initialization_parameters=initialization_parameters,
        )


class SmallMoveSetDefinition(BiRobotSortingDefinition):
    name = "Small Move Set Definition"

    def __init__(self) -> None:
        super().__init__(
            min_damaged_objects=1,
            max_damaged_objects=2,
        )
        self.active_requests = [
            MoveFruitSequence(max_moves=2, weight=1.0),
            SmallMoveFruitSet(
                min_moves=1,
                max_moves=2,
                weight=8.0,
            ),
            TypePriorityRequest(),
            SortDamagedObjectsRequest(),
            AskTypeZoneRequest(),
            AskZoneContentsRequest(),
            AskDamagedObjectsRequest(),
        ]

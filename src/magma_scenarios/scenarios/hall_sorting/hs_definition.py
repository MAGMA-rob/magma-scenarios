import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from magma_core.simulation.data_structures import SituationInit
from magma_core.simulation.state import TaskState
from magma_core.simulation.tasks import InitializationParameters, TaskDefinition

from .attributes import HALLS, OBJECT_TYPES, ROBOT_NAMES
from .hs_tools import HallSortingTool
from .requests import (
    AskHallTypesRequest,
    AskPriorityHallRequest,
    AskTypeHallAssignmentRequest,
    HallPriorityRequest,
    GiveTypeHallAssignmentRequest,
    SortDeliveryCycleRequest,
    SortTypeCycleRequest,
    AdditionalDeliveryInterruptionRequest,
    RedirectObjectInterruptionRequest,
    SmallSortTypeCycleRequest,
    DeliverRecipeRequest,
)
from .rule_renderer import HallSortingRuleRenderer


HALL_CAPACITY = 4
MIN_INSTANCES_PER_TYPE = 1
MAX_INSTANCES_PER_TYPE = 2
RECEPTION_HALL_COUNT = 2


def _sample_objects() -> List[str]:
    objects = []

    for object_type in OBJECT_TYPES:
        instance_count = random.randint(
            MIN_INSTANCES_PER_TYPE,
            MAX_INSTANCES_PER_TYPE,
        )

        objects.extend(
            f"{object_type}_{index}"
            for index in range(1, instance_count + 1)
        )

    random.shuffle(objects)
    return objects


def _sample_hall_roles() -> Tuple[List[str], List[str]]:
    shuffled_halls = HALLS.copy()
    random.shuffle(shuffled_halls)

    reception_halls = shuffled_halls[:RECEPTION_HALL_COUNT]
    storage_halls = shuffled_halls[RECEPTION_HALL_COUNT:]

    return reception_halls, storage_halls


def _split_objects_between_reception_halls(objects: List[str], reception_halls: List[str]) -> Dict[str, List[str]]:
    """
    Split objects between two reception halls.

    Both halls receive at least one object, without exceeding their
    capacity. The split may be asymmetric.
    """
    if len(reception_halls) != RECEPTION_HALL_COUNT:
        raise ValueError(
            f"Expected {RECEPTION_HALL_COUNT} reception halls, "
            f"got {len(reception_halls)}."
        )

    minimum_first_count = max(1, len(objects) - HALL_CAPACITY)
    maximum_first_count = min(HALL_CAPACITY, len(objects) - 1)

    if minimum_first_count > maximum_first_count:
        raise ValueError(
            f"Cannot distribute {len(objects)} objects between "
            f"two halls with capacity {HALL_CAPACITY}."
        )

    first_count = random.randint(
        minimum_first_count,
        maximum_first_count,
    )

    return {
        reception_halls[0]: objects[:first_count],
        reception_halls[1]: objects[first_count:],
    }


def _sample_environment(
    objects: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    reception_halls, storage_halls = _sample_hall_roles()

    if objects is None:
        objects = _sample_objects()

    reception_assignment = _split_objects_between_reception_halls(
        objects=objects,
        reception_halls=reception_halls,
    )

    env_options = {
        hall: reception_assignment.get(hall, [])
        for hall in HALLS
    }

    return env_options, reception_halls, storage_halls


class HallSortingDeliveryDefinition(TaskDefinition):

    name = "Hall Sorting Delivery"
    maniskill_env_id = "4HallSorting-v1"
    Tools_cls = HallSortingTool
    RuleRenderer_cls = HallSortingRuleRenderer

    randomized_config_path = str(
        Path(__file__).resolve().parent / "config.yaml"
    )

    active_requests = [
        GiveTypeHallAssignmentRequest(),
        SortTypeCycleRequest(),
        SortDeliveryCycleRequest(),
        RedirectObjectInterruptionRequest(),
        AdditionalDeliveryInterruptionRequest(),
        # HallPriorityRequest(),
        AskTypeHallAssignmentRequest(),
        DeliverRecipeRequest(),
        AskHallTypesRequest(),
        # AskPriorityHallRequest(),
    ]

    def __init__(self, objects: Optional[List[str]] = None) -> None:
        (env_options, reception_halls, storage_halls) = _sample_environment(
            objects=objects,
        )

        attributes = {
            "known_robots": ROBOT_NAMES.copy(),
            "halls": HALLS.copy(),
            "reception_halls": reception_halls.copy(),
            "storage_halls": storage_halls.copy(),
            "object_classes": OBJECT_TYPES.copy(),
        }

        memory = {
            "memory_list": [
                (
                    "Reception halls contain the products available for "
                    "the current delivery."
                ),
                (
                    "Storage halls are the destinations where products "
                    "must be sorted according to the active rules."
                ),
                (
                    "Only one robot can occupy a hall at a time."
                )
            ]
        }

        situation_init = SituationInit(
            attributes=deepcopy(attributes),
            all_task_attributes={
                "known_robots": ROBOT_NAMES.copy(),
                "halls": HALLS.copy(),
                "reception_halls": HALLS.copy(),
                "storage_halls": HALLS.copy(),
                "object_classes": OBJECT_TYPES.copy(),
            },
            memory=memory,
        )

        starting_state = TaskState()
        starting_state.attributes = deepcopy(attributes)

        starting_state.relations["type_hall"] = {}
        starting_state.properties["delivery_layout"] = deepcopy(
            env_options
        )

        initialization_parameters = InitializationParameters(
            env_options=env_options,
            agent_names=ROBOT_NAMES.copy(),
        )

        super().__init__(
            situation_init=situation_init,
            starting_state=starting_state,
            initialization_parameters=initialization_parameters,
        )


class SmallHallSortingDefinition(HallSortingDeliveryDefinition):
    name = "Small Hall Sorting Definition"

    def __init__(self) -> None:
        objects = [
            f"{object_type}_{index}"
            for object_type in OBJECT_TYPES
            for index in range(1, 3)
        ]
        super().__init__(objects=objects)
        self.active_requests = [
            GiveTypeHallAssignmentRequest(),
            SmallSortTypeCycleRequest(max_objects=2),
            DeliverRecipeRequest(
                max_recipe_types=1,
                max_objects=2,
            ),
            AskTypeHallAssignmentRequest(max_types=1),
            AskHallTypesRequest(),
        ]

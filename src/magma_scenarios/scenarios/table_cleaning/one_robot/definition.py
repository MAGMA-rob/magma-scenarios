from collections import Counter
import random
from pathlib import Path
from typing import Optional

from magma_core.simulation.data_structures import SituationInit
from magma_core.simulation.state import TaskState
from magma_core.simulation.tasks import InitializationParameters, TaskDefinition, TaskMetadata

from ..common.attributes import (
    CLEAN_STATE,
    DIRTY_STATE,
    DISHWASHER_LOCATIONS,
    build_complete_task_attributes,
    build_random_env_start,
    build_task_attributes,
    food,
    generic_rules_simplified,
)
from .clean_tools import TableCleaningTool
from ..rule_renderer import TableCleaningRuleRenderer
from .table_requests import (
    CleanTableRequest,
    DirectSetTableRequest,
    GiveTableAssignmentRequest,
    SetTableRequest,
)


class CleanTableDefinition(TaskDefinition):
    """Continuous single-robot table setup and cleaning scenario."""

    maniskill_env_id = "Cleaning_Table"
    Tools_cls = TableCleaningTool
    RuleRenderer_cls = TableCleaningRuleRenderer
    active_requests = [
        GiveTableAssignmentRequest(),
        CleanTableRequest(),
        SetTableRequest(),
        DirectSetTableRequest(),
    ]

    def __init__(self, nb_dirty: Optional[int] = None) -> None:
        if nb_dirty is not None and (type(nb_dirty) is not int or nb_dirty < 0):
            raise ValueError("nb_dirty must be None or a non-negative integer.")

        table, storage, sampled_dirty = build_random_env_start()
        active_objects = table + storage

        if nb_dirty is not None and nb_dirty > len(active_objects):
            raise ValueError(
                f"nb_dirty cannot exceed the {len(active_objects)} active objects."
            )

        if nb_dirty is None:
            dirty = sampled_dirty
        else:
            missing_table_objects = nb_dirty - len(table)
            if missing_table_objects > 0:
                moved_to_table = random.sample(storage, k=missing_table_objects)
                table.extend(moved_to_table)
                storage = [
                    object_name
                    for object_name in storage
                    if object_name not in moved_to_table
                ]
            dirty = random.sample(table, k=nb_dirty)
        active_objects = table + storage

        object_locations = {
            object_name: (
                "table"
                if object_name in table
                else "food_storage"
                if object_name in food
                else "dish_storage"
            )
            for object_name in active_objects
        }
        object_states = {
            object_name: (
                DIRTY_STATE if object_name in dirty else CLEAN_STATE
            )
            for object_name in active_objects
        }

        attributes = build_task_attributes(DISHWASHER_LOCATIONS)

        memory = {
            "memory_list": generic_rules_simplified.copy(),
        }

        starting_state = TaskState()
        starting_state.attributes = attributes.copy()
        starting_state.memory = {
            key: value.copy()
            for key, value in memory.items()
        }
        starting_state.relations["object_area"] = object_locations
        starting_state.relations["object_state"] = object_states
        starting_state.relations["table_assignment"] = dict(
            Counter(object_name.rsplit("_", 1)[0] for object_name in table)
        )
        starting_state.relations["table_requirements"] = {}
        starting_state.properties["active_objects"] = active_objects.copy()
        starting_state.properties["table_item_count"] = len(table)
        starting_state.properties["table_is_empty"] = len(table) == 0
        starting_state.properties["table_requirements_needs_application"] = False
        starting_state.properties["table_requirements_applications"] = 0
        starting_state.properties["runtime_object_state_exact"] = True

        super().__init__(
            name="Clean Table Definition",
            situation_init=SituationInit(
                attributes=attributes,
                all_task_attributes=build_complete_task_attributes(attributes),
                memory=memory,
            ),
            starting_state=starting_state,
            initialization_parameters=InitializationParameters(
                env_options={
                    "table": table,
                    "storage": storage,
                    "dirty": dirty,
                },
            ),
            randomized_config_path=str(Path(__file__).resolve().parent / "config.yaml"),
            task_metadata=TaskMetadata(
                coaching_hint=(
                    "In this scenario, errors can arise from forgetting to detect the scene before taking an object. Detection is noisy."
                    "Also, take care of the facts that if a put fail, the object is still in gripper and must be try again."
                )
            )
        )

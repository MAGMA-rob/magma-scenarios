from collections import Counter
import random
from pathlib import Path
from typing import Optional

from magma_core.simulation.data_structures import SituationInit
from magma_core.simulation.state import TaskState
from magma_core.simulation.tasks import InitializationParameters, TaskDefinition
from magma_core.simulation.tasks_style import TaskStyle
from magma_core.simulation.tasks import TaskMetadata

from ..common.attributes import (
    ADVANCED_LOCATIONS,
    CLEAN_STATE,
    DIRTY_STATE,
    build_complete_task_attributes,
    build_random_env_start,
    build_task_attributes,
    dishware,
    food,
    generic_rules,
)
from .adct_tools import AdvancedCleaningTools
from ..rule_renderer import TableCleaningRuleRenderer
from .requests import CleanAndStoreRequest, SetTableRequest


def _object_type(object_name: str) -> str:
    return object_name.rsplit("_", 1)[0]


class AdvancedCleanTableDefinition(TaskDefinition):
    """Continuous two-robot table setting and cleaning scenario."""

    maniskill_env_id = "Advanced_Cleaning_Table"
    Tools_cls = AdvancedCleaningTools
    RuleRenderer_cls = TableCleaningRuleRenderer
    active_requests = [
        SetTableRequest(),
        CleanAndStoreRequest(),
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

        assignment_counts = {location: {} for location in ADVANCED_LOCATIONS}
        for object_name in active_objects:
            location = object_locations[object_name]
            object_type = _object_type(object_name)
            state = object_states[object_name]
            state_counts = assignment_counts[location].setdefault(
                object_type,
                {},
            )
            state_counts[state] = state_counts.get(state, 0) + 1

        attributes = build_task_attributes(ADVANCED_LOCATIONS)
        attributes["known_robots"] = ["dish_robot", "table_robot"]
        memory = {
            "memory_list": generic_rules.copy(),
        }

        starting_state = TaskState()
        starting_state.attributes = attributes.copy()
        starting_state.memory = {
            key: value.copy()
            for key, value in memory.items()
        }
        starting_state.relations["object_area"] = object_locations
        starting_state.relations["object_state"] = object_states
        starting_state.relations["object_assignment"] = (
            assignment_counts
        )
        starting_state.properties["active_objects"] = active_objects.copy()
        initial_table_assignment = dict(
            Counter(_object_type(object_name) for object_name in table)
        )
        starting_state.relations["table_assignment"] = (
            initial_table_assignment.copy()
        )
        starting_state.properties["table_contents"] = (
            initial_table_assignment.copy()
        )
        starting_state.properties["table_item_count"] = len(table)
        starting_state.properties["table_is_empty"] = not table

        super().__init__(
            name="Advanced Clean Table Definition",
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
                agent_names=["dish_robot", "table_robot"],
            ),
            task_metadata=TaskMetadata(
                styles=[
                    TaskStyle.CONSTRAINED,
                    TaskStyle.LONG_STAGE,
                    TaskStyle.MULTI_ROBOT,
                ],
                approximal_difficulty="Hard",
                coaching_hint=(
                    "In this scenario, errors can arise from forgetting to detect the scene before taking an object. "
                    "Finally, if we ask to set the table, if objects are already present on the table, you must clean them with dish robot before putting them again on the table. "
                    "Only dish robot has access to the sink, so any object that must be cleaned must be taken by dish robot, place in the dish then cleaned."
                )
            ),
            randomized_config_path=str(Path(__file__).resolve().parent / "config.yaml"),
        )

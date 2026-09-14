import random
from pathlib import Path
from typing import Dict, List, Tuple
from magma_core.simulation.data_structures import SituationInit
from magma_core.simulation.tasks import BaseTask, InitializationParameters, TaskMetadata
from magma_core.simulation.tasks_style import TaskStyle
from ..common.attributes import (
    ADVANCED_LOCATIONS,
    CLEAN_STATE,
    DIRTY_STATE,
    build_complete_task_attributes,
    build_task_attributes,
    dishware,
    food,
    generic_rules,
)
from .adct_stages import (
    ArrangeAndCleanObjectsStage,
    CleanDishesStage,
    WipeTableStage,
)
from .adct_tools import AdvancedCleaningTools
from .trace_planning import build_parallel_plan


def _object_type(obj: str) -> str:
    return obj.rsplit("_", 1)[0]


def _storage_target(obj: str) -> str:
    if obj in food:
        return "food_storage"
    if obj in dishware:
        return "dish_storage"

    raise ValueError(f"Unknown object {obj}")


def _build_random_advanced_env_start(max_on_table: int) -> Tuple[List[str], List[str], List[str]]:
    if max_on_table < 1:
        raise ValueError(f"max_on_table must be at least 1, got {max_on_table}")

    selected_food = random.sample(food, k=3)
    selected_dishware = random.sample(dishware, k=3)
    selected_objects = selected_food + selected_dishware

    table_count = random.randint(1, min(max_on_table, len(selected_objects)))
    table = random.sample(selected_objects, k=table_count)
    storage = [obj for obj in selected_objects if obj not in table]

    dirty_count = random.randint(1, len(table))
    dirty = random.sample(table, k=dirty_count)

    return table, storage, dirty


class BaseAdvancedCleaningTable(BaseTask):
    maniskill_env_id = "Advanced_Cleaning_Table"
    Tools_cls = AdvancedCleaningTools
    randomized_config_path = str(Path(__file__).resolve().parent / "config.yaml")

    def __init__(self, max_on_table: int = 4, env_options: Dict[str, List[str]] | None = None,) -> None:
        super().__init__()

        if env_options is None:
            table, storage, dirty = _build_random_advanced_env_start(max_on_table)
            env_options = {
                "table": table,
                "storage": storage,
                "dirty": dirty,
            }

        agent_names = ["dish_robot", "table_robot"]

        self.initialization_parameters = InitializationParameters(
            env_options=env_options,
            agent_names=agent_names,
        )

        self.obj_on_table = env_options["table"]
        self.obj_on_storage = env_options["storage"]
        self.dirty_obj = env_options.get("dirty", [])
        self.active_objects = self.obj_on_table + self.obj_on_storage

        initial_attributes = {
            **build_task_attributes(ADVANCED_LOCATIONS),
            "robots": agent_names
        }

        self.situation_init = SituationInit(
            attributes=initial_attributes,
            all_task_attributes=build_complete_task_attributes(initial_attributes),
            memory={"memory_list":generic_rules},
        )


class AdvancedCleaningPreset(BaseAdvancedCleaningTable):
    name = "AdvancedCleaning"

    def __init__(self, max_on_table: int = 4) -> None:
        super().__init__(max_on_table=max_on_table)

        dirty_dishes = [
            obj for obj in self.obj_on_table
            if obj in dishware and obj in self.dirty_obj
        ]

        current_locations = {
            obj: "table" if obj in self.obj_on_table else _storage_target(obj)
            for obj in self.active_objects
        }
        current_states = {
            obj: DIRTY_STATE if obj in self.dirty_obj else CLEAN_STATE
            for obj in self.active_objects
        }

        self.stages = []
        if dirty_dishes:
            cleaning_plan = build_parallel_plan(
                stage_count=len(dirty_dishes),
                current_locations={
                    obj: current_locations[obj]
                    for obj in dirty_dishes
                },
                current_states={
                    obj: current_states[obj]
                    for obj in dirty_dishes
                },
                target_locations={
                    obj: "drying_zone"
                    for obj in dirty_dishes
                },
                target_states={
                    obj: CLEAN_STATE
                    for obj in dirty_dishes
                },
            )
            instruction = "Clean all dirty dishes in the sink."
            for minimum, decision_count in enumerate(
                cleaning_plan.decisions_per_stage,
                start=1,
            ):
                stage = CleanDishesStage(
                    minimum=minimum,
                    dishes=dirty_dishes,
                    instruction=instruction,
                    flag_answer=minimum == len(dirty_dishes),
                )
                stage.target_tool_calls = decision_count
                stage.max_tool_calls = max(8, decision_count)
                self.stages.append(stage)
                instruction = "none"

        locations_after_cleaning = current_locations.copy()
        states_after_cleaning = current_states.copy()
        for obj in dirty_dishes:
            locations_after_cleaning[obj] = "drying_zone"
            states_after_cleaning[obj] = CLEAN_STATE

        required_state_counts: Dict[str, Dict[str, Dict[int, int]]] = {
            location: {}
            for location in ADVANCED_LOCATIONS
        }
        target_locations: Dict[str, str] = {}
        target_states: Dict[str, int] = {}
        for obj in self.active_objects:
            target_location = (
                "trashcan"
                if obj in food and obj in self.dirty_obj
                else _storage_target(obj)
            )
            target_state = (
                DIRTY_STATE
                if obj in food and obj in self.dirty_obj
                else CLEAN_STATE
            )
            target_locations[obj] = target_location
            target_states[obj] = target_state
            state_counts = required_state_counts[target_location].setdefault(
                _object_type(obj),
                {},
            )
            state_counts[target_state] = state_counts.get(target_state, 0) + 1

        initial_correct = sum(
            locations_after_cleaning[obj] == target_locations[obj]
            and states_after_cleaning[obj] == target_states[obj]
            for obj in self.active_objects
        )
        stage_count = len(self.active_objects) - initial_correct
        plan = build_parallel_plan(
            stage_count=stage_count,
            current_locations=locations_after_cleaning,
            current_states=states_after_cleaning,
            target_locations=target_locations,
            target_states=target_states,
        )
        decisions_per_stage = list(plan.decisions_per_stage)
        # The planner reserves one final decision for the completion message.
        # This preset emits that message only after the following wipe stage.
        decisions_per_stage[-1] -= 1

        instruction = (
            "Store the clean dishware and clear the table by putting every "
            "object in its correct location."
        )
        for minimum, decision_count in zip(
            range(initial_correct + 1, len(self.active_objects) + 1),
            decisions_per_stage,
        ):
            stage = ArrangeAndCleanObjectsStage(
                minimum=minimum,
                required_state_counts=required_state_counts,
                instruction=instruction,
                flag_answer=False,
            )
            stage.target_tool_calls = decision_count
            stage.max_tool_calls = max(
                stage.get_max_tool_calls() or 0,
                decision_count,
            )
            self.stages.append(stage)
            instruction = "none"

        self.stages.append(
            WipeTableStage(
                instruction="Now wipe the empty table."
            )
        )

        self.task_metadata = TaskMetadata(
            styles=[
                TaskStyle.CONSTRAINED,
                TaskStyle.LONG_STAGE,
                TaskStyle.MULTI_ROBOT,
            ],
            approximal_difficulty=(
                "Hard"
                if dirty_dishes or len(self.obj_on_table) >= 3
                else "Medium"
            ),
            coaching_hint=(
                "Only dish robot has access to the sink, so any object that must be cleaned must be taken by dish robot, place in the dish then cleaned."
            )
        )

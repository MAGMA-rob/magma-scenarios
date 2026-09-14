import random
from pathlib import Path
from typing import Dict, List, Tuple
from magma_core.simulation.stage import ConstraintBaseStage
from magma_core.simulation.tasks import BaseTask
from magma_core.simulation.data_structures import SituationInit
from ..common.attributes import (
    DISHWASHER_LOCATIONS,
    build_complete_task_attributes,
    build_task_attributes,
    dishware,
    food,
    generic_rules_simplified,
)
from .clean_stage import PlaceObjectsStage, WipeStage
from .clean_tools import TableCleaningTool

def _object_type(obj: str) -> str:
    return obj.rsplit("_", 1)[0]

def _active_types(objects: List[str]) -> List[str]:
    return sorted({_object_type(obj) for obj in objects})

def _empty_typed_assignment( locations: List[str], object_types: List[str]) -> Dict[str, Dict[str, int]]:
    return {
        location: {obj_type: 0 for obj_type in object_types}
        for location in locations
    }

def _storage_target(obj: str) -> str:
    if obj in food:
        return "food_storage"
    if obj in dishware:
        return "dish_storage"

    raise ValueError(f"Unknown object {obj}")


def _dirty_target(obj: str) -> str:
    if obj in food:
        return "trashcan"
    if obj in dishware:
        return "washing_machine"

    raise ValueError(f"Unknown object {obj}")

def _build_recipe_counts(recipe_objects: List[str]) -> Dict[str, int]:
    recipe_counts: Dict[str, int] = {}

    for obj in recipe_objects:
        obj_type = _object_type(obj)
        recipe_counts[obj_type] = recipe_counts.get(obj_type, 0) + 1

    return recipe_counts


def _recipe_to_text(recipe: Dict[str, int]) -> str:
    parts = []

    for obj_type, count in recipe.items():
        if count <= 0:
            continue

        if count :
            parts.append(f"{count} {obj_type}")

    return ", ".join(parts)

def _build_random_remove_env_start(max_on_table: int) -> Tuple[List[str], List[str], List[str]]:
    if max_on_table < 1:
        raise ValueError(f"max_on_table must be at least 1, got {max_on_table}")

    selected_food = random.sample(food, k=3)
    selected_dishware = random.sample(dishware, k=3)
    selected_objects = selected_food + selected_dishware

    table_count = random.randint(1, min(max_on_table, len(selected_objects)))
    table = random.sample(selected_objects, k=table_count)
    storage = [obj for obj in selected_objects if obj not in table]

    dirty_count = random.randint(0, len(table))
    dirty = random.sample(table, k=dirty_count)

    return table, storage, dirty


def _build_table_cleaning_assignments(table: List[str], storage: List[str], dirty: List[str], recipe: Dict[str, int] | None = None) -> Tuple[Dict[str, Dict[str, int]], Dict[str, str]]:
    active_objects = table + storage
    active_types = _active_types(active_objects)

    assignment = _empty_typed_assignment(DISHWASHER_LOCATIONS, active_types)
    fixed_targets_by_object: Dict[str, str] = {}

    recipe_counts = dict(recipe or {})

    for obj in active_objects:
        obj_type = _object_type(obj)

        if obj in dirty:
            target = _dirty_target(obj)
            fixed_targets_by_object[obj] = target

        elif recipe_counts.get(obj_type, 0) > 0:
            target = "table"
            recipe_counts[obj_type] -= 1

        else:
            target = _storage_target(obj)

        assignment[target][obj_type] += 1

    return assignment, fixed_targets_by_object

def _count_initial_correct_objects(
    table: List[str],
    storage: List[str],
    dirty: List[str],
    required_type_counts: Dict[str, Dict[str, int]],
    fixed_targets_by_object: Dict[str, str],
) -> int:
    initial_location = {}

    for obj in table:
        initial_location[obj] = "table"

    for obj in storage:
        initial_location[obj] = _storage_target(obj)

    correct = 0
    remaining_assignment = {
        location: dict(type_counts)
        for location, type_counts in required_type_counts.items()
    }

    for obj, location in initial_location.items():
        obj_type = _object_type(obj)

        if obj in fixed_targets_by_object:
            if fixed_targets_by_object[obj] == location:
                correct += 1
            continue

        if remaining_assignment.get(location, {}).get(obj_type, 0) > 0:
            correct += 1
            remaining_assignment[location][obj_type] -= 1

    return correct

class BaseCleaningTable(BaseTask):
    maniskill_env_id = "Cleaning_Table"
    Tools_cls = TableCleaningTool
    randomized_config_path = str(Path(__file__).resolve().parent / "config.yaml")

    def __init__(self, max_on_table: int = 4, env_options: Dict[str, List[str]] | None = None) -> None:
        super().__init__()

        if env_options is None:
            table, storage, dirty = _build_random_remove_env_start(max_on_table)
            env_options = {
                "table": table,
                "storage": storage,
                "dirty": dirty,
            }
        self.initialization_parameters.env_options = env_options

        self.obj_on_table = env_options["table"]
        self.obj_on_storage = env_options["storage"]
        self.dirty_obj = env_options.get("dirty", [])
        self.active_objects = self.obj_on_table + self.obj_on_storage

        initial_attributes = build_task_attributes(DISHWASHER_LOCATIONS)
        self.situation_init = SituationInit(
            attributes=initial_attributes,
            all_task_attributes=build_complete_task_attributes(initial_attributes),
            memory={"memory_list": generic_rules_simplified},
        )


class TableCleaningPreset(BaseCleaningTable):
    def __init__(self, max_on_table: int = 2) -> None:
        super().__init__(max_on_table=max_on_table)

        required_type_counts, fixed_targets_by_object = _build_table_cleaning_assignments(
            table=self.obj_on_table,
            storage=self.obj_on_storage,
            dirty=self.dirty_obj,
        )

        self.stages = []

        instruction = "Remove all objects from the table."

        total_targets = len(self.active_objects)
        already_correct = len(self.obj_on_storage)

        for n in range(already_correct + 1, total_targets + 1):
            self.stages.append(
                PlaceObjectsStage(
                    n=n,
                    required_type_counts=required_type_counts,
                    fixed_targets_by_object=fixed_targets_by_object,
                    instruction=instruction,
                    flag_answer=n==total_targets,
                )
            )
            instruction="none"

        self.stages.append(
            WipeStage(
                instruction="Now wipe the table.",
                flag_answer=True,
            )
        )


class SetTablePreset(BaseCleaningTable):
    name = "SetTable"

    def __init__(self, max_on_table: int = 2) -> None:
        super().__init__(max_on_table=max_on_table)

        available_objects = [
            obj for obj in self.active_objects
            if obj not in self.dirty_obj
        ]

        recipe_size = random.randint(1, min(3, len(available_objects)))
        recipe_objects = random.sample(available_objects, k=recipe_size)
        recipe = _build_recipe_counts(recipe_objects)

        required_type_counts, fixed_targets_by_object = _build_table_cleaning_assignments(
            table=self.obj_on_table,
            storage=self.obj_on_storage,
            dirty=self.dirty_obj,
            recipe=recipe,
        )

        recipe_text = _recipe_to_text(recipe)

        self.stages = []

        instruction = f"Set {recipe_text} on the table please."

        total_targets = len(self.active_objects)

        initial_correct = _count_initial_correct_objects(
            table=self.obj_on_table,
            storage=self.obj_on_storage,
            dirty=self.dirty_obj,
            required_type_counts=required_type_counts,
            fixed_targets_by_object=fixed_targets_by_object,
        )

        for n in range(initial_correct + 1, total_targets + 1):
            self.stages.append(
                PlaceObjectsStage(
                    n=n,
                    required_type_counts=required_type_counts,
                    fixed_targets_by_object=fixed_targets_by_object,
                    instruction=instruction,
                    flag_answer=n == total_targets,
                )
            )
            instruction = "none"

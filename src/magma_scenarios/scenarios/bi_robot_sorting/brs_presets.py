# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
import copy
import random
from .brs_attributes import perturb_assignment
from pathlib import Path

from magma_core.simulation.tasks import BaseTask, InitializationParameters
from magma_core.simulation.data_structures import SituationInit

from .brs_stages import (
    AskDamagedObjectsStage,
    BRSAccessConstraintStage,
    RecipeObjectsStage,
    SortDamagedObjectsStage,
)
from .brs_attributes import ZONES, OBJECT_TYPES, ROBOTS, sorting_objects
from .brs_tools import BiRobotTools
from .brs_attributes import build_random_env_start


def _empty_typed_assignment() -> dict[str, dict[str, int]]:
    return {
        zone: {obj_type: 0 for obj_type in OBJECT_TYPES}
        for zone in ZONES
    }


def _all_objects() -> list[str]:
    return [
        obj
        for objects in sorting_objects.values()
        for obj in objects
    ]


def _object_type(obj: str) -> str:
    return obj.rsplit("_", 1)[0]


def _build_damaged_assignment(damaged_zone: str, damaged_objs: list[str]) -> dict[str, list[str]]:
    return {damaged_zone: damaged_objs}

def _build_goal_assignment(damaged_assignment: dict[str, list[str]], recipe_zone: str, rest_zone: str) -> dict[str, dict[str, int]]:
    assignment = _empty_typed_assignment()

    for zone, objects in damaged_assignment.items():
        for obj in objects:
            assignment[zone][_object_type(obj)] += 1

    recipe_size = random.randint(2, 3)

    for _ in range(recipe_size):
        available_types = [
            obj_type for obj_type in OBJECT_TYPES
            if sum(assignment[zone][obj_type] for zone in ZONES) < len(sorting_objects[obj_type])
        ]

        obj_type = random.choice(available_types)
        assignment[recipe_zone][obj_type] += 1

    for obj_type in OBJECT_TYPES:
        missing_count = len(sorting_objects[obj_type]) - sum(
            assignment[zone][obj_type]
            for zone in ZONES
        )

        assignment[rest_zone][obj_type] += missing_count

    return assignment



def _build_env_options_from_assignment(assignment: dict[str, dict[str, int]], damaged_assignment: dict[str, list[str]],) -> dict[str, list[str]]:
    env_options = {zone: [] for zone in ZONES}

    damaged_objs = {
        obj
        for objects in damaged_assignment.values()
        for obj in objects
    }

    remaining_by_type = {
        obj_type: [
            obj for obj in sorting_objects[obj_type]
            if obj not in damaged_objs
        ]
        for obj_type in OBJECT_TYPES
    }

    for damaged_zone, damaged_objects in damaged_assignment.items():
        for obj in damaged_objects:
            obj_type = _object_type(obj)

            if assignment[damaged_zone][obj_type] > 0:
                target_zone = damaged_zone
            else:
                candidate_zones = [
                    zone for zone in ZONES
                    if assignment[zone][obj_type] > 0
                ]
                target_zone = random.choice(candidate_zones)

            env_options[target_zone].append(obj)
            assignment[target_zone][obj_type] -= 1

    for zone, type_counts in assignment.items():
        for obj_type, count in type_counts.items():
            for _ in range(count):
                obj = remaining_by_type[obj_type].pop()
                env_options[zone].append(obj)

    env_options["damaged_objs"] = list(damaged_objs)

    return env_options

def _split_goal_by_condition(
    goal_assignment: dict[str, dict[str, int]],
    damaged_object_assignment: dict[str, list[str]],
) -> tuple[
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
]:
    intact_assignment = copy.deepcopy(goal_assignment)
    damaged_assignment = _empty_typed_assignment()

    for zone, object_names in damaged_object_assignment.items():
        for object_name in object_names:
            object_type = _object_type(object_name)

            damaged_assignment[zone][object_type] += 1
            intact_assignment[zone][object_type] -= 1

            if intact_assignment[zone][object_type] < 0:
                raise ValueError(
                    f"Invalid damaged assignment for "
                    f"{object_name!r} in {zone!r}."
                )

    return intact_assignment, damaged_assignment

class BaseBRS(BaseTask):
    maniskill_env_id = "BiRobotSorting-v1"
    Tools_cls = BiRobotTools
    randomized_config_path = str(Path(__file__).resolve().parent / "config.yaml")

    situation_init = SituationInit(
        attributes={
            "known_robots": ROBOTS,
            "locations": ZONES,
            "object_classes": OBJECT_TYPES,
        }
    )

    def __init__(self, max_nb_per_zone: int = 4, env_options: dict | None = None) -> None:
        super().__init__()

        if env_options is None:
            left_zone, mutual_zone, right_zone, damaged_objs = build_random_env_start(
                max_per_zone=max_nb_per_zone
            )
            env_options = {
                "left_zone": left_zone,
                "mutual_zone": mutual_zone,
                "right_zone": right_zone,
                "damaged_objs": damaged_objs,
            }
        else:
            env_options = dict(env_options)
            env_options.setdefault("damaged_objs", [])

        self.left_zone = env_options.get("left_zone", [])
        self.mutual_zone = env_options.get("mutual_zone", [])
        self.right_zone = env_options.get("right_zone", [])
        self.damaged_objs = env_options.get("damaged_objs", [])

        self.initialization_parameters = InitializationParameters(
            env_options=env_options,
            agent_names=ROBOTS,
        )

        self.object_locations = {}

        for obj in self.left_zone:
            self.object_locations[obj] = "left_zone"

        for obj in self.mutual_zone:
            self.object_locations[obj] = "mutual_zone"

        for obj in self.right_zone:
            self.object_locations[obj] = "right_zone"


class PrepareRecipePreset(BaseBRS):
    name = "Prepare Recipe With Damaged Sorting"

    def __init__(self, nb_misplaced: int = 2) -> None:
        all_objects = _all_objects()

        recipe_zone = random.choice(ZONES)
        remaining_zones = [zone for zone in ZONES if zone != recipe_zone]

        damaged_zone = random.choice(remaining_zones)
        rest_zone = [zone for zone in remaining_zones if zone != damaged_zone][0]

        damaged_objs = random.sample(all_objects, k=random.randint(1, 3))

        self.recipe_zone = recipe_zone
        self.damaged_zone = damaged_zone
        self.rest_zone = rest_zone
        self.damaged_objs = damaged_objs

        self.damaged_object_assignment = (
            _build_damaged_assignment(
                damaged_zone=damaged_zone,
                damaged_objs=damaged_objs,
            )
        )

        self.goal_assignment = _build_goal_assignment(
            damaged_assignment=self.damaged_object_assignment,
            recipe_zone=recipe_zone,
            rest_zone=rest_zone,
        )

        (
        self.intact_goal_assignment,
        self.damaged_goal_assignment,
        ) = _split_goal_by_condition(
            goal_assignment=self.goal_assignment,
            damaged_object_assignment=(
                self.damaged_object_assignment
            ),
        )

        initial_assignment, actual_misplaced = perturb_assignment(
            assignment=self.goal_assignment,
            max_moves=nb_misplaced,
        )

        env_options = _build_env_options_from_assignment(
            assignment=copy.deepcopy(initial_assignment),
            damaged_assignment=self.damaged_object_assignment,
        )

        super().__init__(env_options=env_options)

        recipe_parts = [
            f"{count} {obj_type}"
            for obj_type, count in self.goal_assignment[recipe_zone].items()
            if count > 0
        ]
        recipe_str = ", ".join(recipe_parts)

        instruction = (
            f"Prepare the recipe in {recipe_zone}. "
            f"The recipe must contain: {recipe_str}. "
            f"Damaged objects must be placed in {self.damaged_zone}. "
            f"All remaining objects must be placed in {rest_zone}."
        )

        self.stages = [
            BRSAccessConstraintStage(),
            AskDamagedObjectsStage(
                damaged_objects=damaged_objs,
                object_locations=self.object_locations,
            ),
        ]

        total_targets = len(all_objects)
        start_n = len(all_objects) - actual_misplaced + 1

        for n in range(start_n, total_targets + 1):
            self.stages.append(
                RecipeObjectsStage(
                    n=n,
                    progress=n - start_n + 1,
                    progress_total=actual_misplaced,
                    intact_assignment=self.intact_goal_assignment,
                    damaged_assignment=self.damaged_goal_assignment,
                    damaged_objects=damaged_objs.copy(),
                    instruction=(
                        instruction
                        if n == start_n
                        else "none"
                    ),
                ))

class SortFruitsByTypePreset(BaseBRS):
    name = "Sort Fruits By Type"

    def __init__(self, nb_misplaced: int = 2) -> None:
        all_objects = _all_objects()

        shuffled_zones = ZONES.copy()
        random.shuffle(shuffled_zones)

        type_to_zone = {
            obj_type: zone
            for obj_type, zone in zip(OBJECT_TYPES, shuffled_zones)
        }

        self.goal_assignment = _empty_typed_assignment()

        for obj_type, zone in type_to_zone.items():
            self.goal_assignment[zone][obj_type] = len(sorting_objects[obj_type])

        self.damaged_object_assignment: dict[str, list[str]] = {}

        self.intact_goal_assignment = copy.deepcopy(self.goal_assignment)
        self.damaged_goal_assignment = (_empty_typed_assignment())

        initial_assignment, actual_misplaced = perturb_assignment(
            assignment=self.goal_assignment,
            max_moves=nb_misplaced,
        )

        env_options = _build_env_options_from_assignment(
            assignment=copy.deepcopy(initial_assignment),
            damaged_assignment=self.damaged_object_assignment,
        )

        super().__init__(env_options=env_options)

        instruction = (
            "Sort the fruits by type. "
            + " ".join(
                f"Place all {obj_type} objects in {zone}."
                for obj_type, zone in type_to_zone.items()
            )
        )

        self.stages = [
            BRSAccessConstraintStage(),
        ]

        total_targets = len(all_objects)
        start_n = total_targets - actual_misplaced + 1

        for n in range(start_n, total_targets + 1):
            self.stages.append(
               RecipeObjectsStage(
                n=n,
                progress=n - start_n + 1,
                progress_total=actual_misplaced,
                intact_assignment=self.intact_goal_assignment,
                damaged_assignment=self.damaged_goal_assignment,
                damaged_objects=[],
                instruction=(
                    instruction
                    if n == start_n
                    else "none"
                ),
            ))

class TestSortDamagedObjectsPreset(BaseBRS):
    name = "Test Sort Damaged Objects"

    def __init__(self) -> None:
        damaged_objects = ["banana_0", "apple_0"]
        target_zone = "mutual_zone"

        env_options = {
            "left_zone": [
                "banana_0",
                "banana_1",
                "apple_1",
                "orange_0",
            ],
            "mutual_zone": ["orange_2"],
            "right_zone": [
                "apple_0",
                "banana_2",
                "apple_2",
                "orange_1",
            ],
            "damaged_objs": damaged_objects,
        }

        super().__init__(env_options=env_options)

        instruction = (
            "Place all damaged objects in the mutual zone. "
            "Use arm1 for the damaged object in the left zone and "
            "arm2 for the damaged object in the right zone."
        )

        self.stages = [
            SortDamagedObjectsStage(
                damaged_objects=damaged_objects,
                target_zone=target_zone,
                minimum=minimum,
                instruction=instruction if minimum == 1 else "none",
                last=(minimum == len(damaged_objects)),
            )
            for minimum in range(1, len(damaged_objects) + 1)
        ]

import copy
import random
from pathlib import Path
from typing import Dict
from magma_core.simulation.tasks import BaseTask, InitializationParameters
from magma_core.simulation.data_structures import SituationInit
from .attributes import HALLS, OBJECT_TYPES, OBJECT_NAMES, ROBOT_NAMES, build_random_env_start
from .hs_tools import HallSortingTool
from .hs_stages import TypedPlacementStage


def _empty_typed_assignment() -> dict[str, dict[str, int]]:
    return {
        hall: {obj_type: 0 for obj_type in OBJECT_TYPES}
        for hall in HALLS
    }


def _build_env_options_from_assignment(assignment: dict[str, dict[str, int]]) -> dict[str, list[str]]:
    env_options = {hall: [] for hall in HALLS}

    remaining_by_type = {
        obj_type: [
            obj for obj in OBJECT_NAMES
            if obj.startswith(obj_type + "_")
        ]
        for obj_type in OBJECT_TYPES
    }

    for obj_type in OBJECT_TYPES:
        random.shuffle(remaining_by_type[obj_type])

    for hall, type_counts in assignment.items():
        for obj_type, count in type_counts.items():
            for _ in range(count):
                if not remaining_by_type[obj_type]:
                    raise ValueError(
                        f"No remaining {obj_type} object to instantiate."
                    )

                obj = remaining_by_type[obj_type].pop()
                env_options[hall].append(obj)

    return env_options


def _perturb_assignment_with_swaps(assignment: dict[str, dict[str, int]], max_nb_swaps: int) -> tuple[dict[str, dict[str, int]], int]:
    if max_nb_swaps < 1:
        raise ValueError(
            f"max_nb_swaps must be at least 1, got {max_nb_swaps}"
        )

    perturbed = copy.deepcopy(assignment)

    actual_swaps = 0
    increased = set()
    decreased = set()

    for _ in range(max_nb_swaps):
        possible_swaps = []

        for hall_a in HALLS:
            for hall_b in HALLS:
                if hall_a == hall_b:
                    continue

                for type_a in OBJECT_TYPES:
                    if perturbed[hall_a][type_a] <= 0:
                        continue

                    if (hall_a, type_a) in increased:
                        continue

                    if (hall_b, type_a) in decreased:
                        continue

                    for type_b in OBJECT_TYPES:
                        if type_a == type_b:
                            continue

                        if perturbed[hall_b][type_b] <= 0:
                            continue

                        if (hall_b, type_b) in increased:
                            continue

                        if (hall_a, type_b) in decreased:
                            continue

                        possible_swaps.append((hall_a, hall_b, type_a, type_b))

        if not possible_swaps:
            break

        hall_a, hall_b, type_a, type_b = random.choice(possible_swaps)

        perturbed[hall_a][type_a] -= 1
        perturbed[hall_b][type_a] += 1

        perturbed[hall_b][type_b] -= 1
        perturbed[hall_a][type_b] += 1

        decreased.add((hall_a, type_a))
        increased.add((hall_b, type_a))

        decreased.add((hall_b, type_b))
        increased.add((hall_a, type_b))

        actual_swaps += 1

    return perturbed, actual_swaps


class BaseHallSorting(BaseTask):
    maniskill_env_id = "4HallSorting-v1"
    Tools_cls = HallSortingTool
    randomized_config_path = str(Path(__file__).resolve().parent / "config.yaml")

    situation_init = SituationInit(
        attributes={
            "known_robots": ROBOT_NAMES,
            "halls": HALLS,
            "object_classes": OBJECT_TYPES,
        }
    )

    def __init__(self, max_per_zone: int = 3, env_options: Dict | None = None) -> None:
        super().__init__()

        if env_options is None:
            env_options = build_random_env_start(max_per_zone=max_per_zone)

        self.initialization_parameters = InitializationParameters(
            env_options=env_options,
            agent_names=ROBOT_NAMES,
        )

        self.halls = HALLS
        self.object_types = OBJECT_TYPES

class BalancedTypesByHallPreset(BaseHallSorting):
    name = "Balanced Types By Hall"

    def __init__(self, max_nb_swaps: int = 1) -> None:
        if max_nb_swaps < 1:
            raise ValueError(
                f"max_nb_swaps must be at least 1, got {max_nb_swaps}"
            )

        self.goal_assignment = {
            hall: {obj_type: 1 for obj_type in OBJECT_TYPES}
            for hall in HALLS
        }

        initial_assignment, actual_swaps = _perturb_assignment_with_swaps(
            assignment=self.goal_assignment,
            max_nb_swaps=max_nb_swaps,
        )

        env_options = _build_env_options_from_assignment(
            assignment=copy.deepcopy(initial_assignment),
        )

        super().__init__(
            max_per_zone=4,
            env_options=env_options,
        )

        instruction = (
            "Balance the halls by object type. "
            "Each hall must contain one book, one pen, one backpack, and one package."
        )

        self.stages = []

        total_targets = len(HALLS) * len(OBJECT_TYPES)

        actual_misplaced = actual_swaps * 2
        start_n = total_targets - actual_misplaced + 1

        for n in range(start_n, total_targets + 1):
            self.stages.append(
                TypedPlacementStage(
                    n=n,
                    assignment=self.goal_assignment,
                    instruction=instruction if n == start_n else "none",
                )
            )

class SortTypesByHallPreset(BaseHallSorting):
    name = "Sort Object Types By Hall"

    def __init__(self,max_per_zone: int = 4, max_nb_swaps: int = 1) -> None:
        if max_per_zone < 1 or max_per_zone > 4:
            raise ValueError(
                f"max_per_zone must be between 1 and 4, got {max_per_zone}"
            )

        if max_nb_swaps < 1:
            raise ValueError(
                f"max_nb_swaps must be at least 1, got {max_nb_swaps}"
            )

        random_halls = HALLS.copy()
        random.shuffle(random_halls)

        self.type_assignment = {
            obj_type: hall
            for obj_type, hall in zip(OBJECT_TYPES, random_halls)
        }

        self.goal_assignment = _empty_typed_assignment()

        for obj_type, hall in self.type_assignment.items():
            self.goal_assignment[hall][obj_type] = max_per_zone

        initial_assignment, actual_swaps = _perturb_assignment_with_swaps(
            assignment=self.goal_assignment,
            max_nb_swaps=max_nb_swaps,
        )

        env_options = _build_env_options_from_assignment(assignment=copy.deepcopy(initial_assignment))

        super().__init__(max_per_zone=max_per_zone, env_options=env_options)

        instruction = (
            "Sort all objects by type. "
            + " ".join(
                f"Place all {obj_type} objects in {hall}."
                for obj_type, hall in self.type_assignment.items()
            )
        )

        self.stages = []

        total_targets = max_per_zone * len(OBJECT_TYPES)

        actual_misplaced = actual_swaps * 2
        start_n = total_targets - actual_misplaced + 1

        for n in range(start_n, total_targets + 1):
            self.stages.append(
                TypedPlacementStage(
                    n=n,
                    assignment=self.goal_assignment,
                    instruction=instruction if n == start_n else "none",
                )
            )

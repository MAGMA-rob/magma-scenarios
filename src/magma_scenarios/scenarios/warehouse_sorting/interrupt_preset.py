# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.simulation.tasks import BaseTask
from magma_core.simulation.stage import ConstraintBaseStage
from magma_core.simulation.data_structures import SituationInit, UserInstruction
from pathlib import Path
import random

from .tools import WithoutManufacturingOrder
from .warehouse_stages import build_object_to_zone_stages
from .att import OBJECTS, AREAS
ALL_TASK_ATTRIBUTES = {
    "objects" : OBJECTS,
    "target_areas" : AREAS
}

class InterruptPreset(BaseTask):
    """
    Debug task that interrupts a two-object sorting cycle with a priority cycle.
    """
    name : str = "Warehouse sorting interrupted cycle"
    maniskill_env_id : str = "SortingCubesWarehouse-v1"

    Tools_cls = WithoutManufacturingOrder

    randomized_config_path = str(Path(__file__).parent.joinpath("config.yaml"))

    def __init__(
            self,
            nb_of_object : int = 3,
            nb_of_area : int = 5,
        ) -> None:
        """
        Ask for a sorting cycle and interrupt it after its first completed object.
        """
        super().__init__()

        if nb_of_object < 3 or nb_of_object > len(OBJECTS):
            raise ValueError(
                f"InterruptPreset requires between 3 and {len(OBJECTS)} objects, "
                f"got {nb_of_object}."
            )
        if nb_of_area <= 0 or nb_of_area > len(AREAS):
            raise ValueError(
                f"InterruptPreset requires between 1 and {len(AREAS)} areas, "
                f"got {nb_of_area}."
            )

        known_objects = OBJECTS[:nb_of_object]
        known_areas = AREAS[:nb_of_area]

        task_attributes = {
            "objects": known_objects,
            "target_areas": known_areas,
        }

        self.situation_init = SituationInit(
            attributes=task_attributes,
            all_task_attributes=ALL_TASK_ATTRIBUTES,
        )

        self.stages = []

        obj_to_sort = known_objects.copy()
        random.shuffle(obj_to_sort)

        assignment = {
            obj_to_sort[0]: random.choice(known_areas),
            obj_to_sort[1]: random.choice(known_areas),
        }
        constraint = (
            f"Remember that {obj_to_sort[0]} must go to {assignment[obj_to_sort[0]]} "
            f"and {obj_to_sort[1]} must go to {assignment[obj_to_sort[1]]}."
        )
        self.stages.append(ConstraintBaseStage(constraint))

        cycle_stages = build_object_to_zone_stages(
            assignment=assignment,
            instruction=UserInstruction(f"Hey, sort {obj_to_sort[0]} and {obj_to_sort[1]}."),
            known_areas=known_areas.copy(),
            flag_answer=True,
        )

        interrupt_assignment = {
            obj_to_sort[2]: random.choice(known_areas),
        }
        interrupt_instruction = UserInstruction(
            f"Stop the current sorting cycle and sort {obj_to_sort[2]} into "
            f"{interrupt_assignment[obj_to_sort[2]]}. This task has priority. Finish the current after."
        )
        interrupt_stage = build_object_to_zone_stages(
            assignment=interrupt_assignment,
            known_areas=known_areas.copy(),
            instruction=interrupt_instruction,
            reset_at_end=False,
            flag_answer=False,
        )

        self.stages.append(cycle_stages[0])
        self.stages.extend(interrupt_stage + cycle_stages[1:])

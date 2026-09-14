# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from pathlib import Path

from magma_core.simulation.tasks import TaskDefinition
from magma_core.simulation.state import TaskState
from magma_core.simulation.data_structures import SituationInit

from .attributes import all_clothes, all_detergents
from .load_tool import LaunchTool
from .laundry_constraints import CLOTHE_DETERGENT_KEY
from .laundry_request import (
    AssignClotheDetergentRequest,
    AskLaundryByDetergentRequest,
    AskLaundryRequest,
    AskClothesDetergentRequest,
    AskClothesDetergentRequestInverse,
    LaundryInterruptionRequest,
)
from .rule_renderer import LaundryRuleRenderer


RANDOMIZED_CONFIG_PATH = str(Path(__file__).resolve().parent / "laundry.yaml")


class LaundryDefinition(TaskDefinition):
    maniskill_env_id = "Laundry-v1"
    randomized_config_path = RANDOMIZED_CONFIG_PATH
    Tools_cls = LaunchTool
    RuleRenderer_cls = LaundryRuleRenderer

    active_requests = [
        AssignClotheDetergentRequest(),
        LaundryInterruptionRequest(),
        AskLaundryRequest(),
        AskLaundryByDetergentRequest(),
        AskClothesDetergentRequest(),
        AskClothesDetergentRequestInverse()
    ]

    def __init__(self) -> None:
        super().__init__(
            name="Laundry definition",
            situation_init=SituationInit(
                attributes={
                    "clothes": all_clothes.copy(),
                    "detergents": all_detergents.copy(),
                    "known_robots": ["default"],
                },
                memory={"memory_list":[
                    "To wash clothes, I need to put them inside the wash-machine, add detergents and then use 'wash'.",
                    "Detergent must always be put last in the wash-machine",
                    "Different detergents must not mixed in the same wash."
                ]}
            ),
        )

        self.starting_state = TaskState()
        self.starting_state.attributes = {
            "clothes": all_clothes.copy(),
            "detergents": all_detergents.copy(),
            "known_robots": ["default"],
        }
        self.starting_state.relations = {
            CLOTHE_DETERGENT_KEY: {},
        }

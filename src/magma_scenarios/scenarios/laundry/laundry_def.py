# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from pathlib import Path

from magma_core.base.tasks import TaskDefinition
from magma_core.base.state import TaskState

from .attributes import all_clothes, all_detergents, all_categories
from .load_tool import LaunchTool
from .laundry_constraints import CLOTHE_DETERGENT_KEY
from .laundry_request import (
    AssignClotheCategoryRequest,
    AssignClotheDetergentRequest,
    AskLaundryByDetergentRequest,
    AskDirectLaundryRequest,
    AskLaundryRequest,
    AskClothesCategoryRequest
)


RANDOMIZED_CONFIG_PATH = str(Path(__file__).resolve().parent / "laundry.yaml")


class LaundryDefinition(TaskDefinition):
    env_id = "Laundry-v1"
    randomized_config_path = RANDOMIZED_CONFIG_PATH
    Tools_cls = LaunchTool

    active_requests = [
        AssignClotheCategoryRequest(),
        AssignClotheDetergentRequest(),
        AskDirectLaundryRequest(),
        AskLaundryRequest(),
        AskLaundryByDetergentRequest(),
        AskClothesCategoryRequest(),
    ]

    def __init__(self) -> None:
        super().__init__(
            name="Laundry definition",
            randomized_config_path=RANDOMIZED_CONFIG_PATH,
        )

        self.starting_state = TaskState()
        self.starting_state.attributes = {
            "clothes": all_clothes.copy(),
            "detergents": all_detergents.copy(),
            "categories": all_categories.copy(),
        }
        self.starting_state.relations = {
            CLOTHE_DETERGENT_KEY: {},
        }

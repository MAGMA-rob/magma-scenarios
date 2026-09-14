# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from collections import Counter
from pathlib import Path
import random

from magma_core.simulation.data_structures import SituationInit
from magma_core.simulation.tasks import BaseBenchmarkTask, BaseTask, InitializationParameters

from .attributes import _sample_table_objects, att
from .delivery_stages import AtLeastPackageCompletedStage
from .delivery_tools import CycleTool

class BenchDeliveryTask(BaseBenchmarkTask):
    """
    Class for the Delivery Benchmark Task. The idea is to have a model which must follow a recipe to complete a delivery box.
    It must reason about multiple constraints.
    """

    name : str = "Make devlivery [benchmark]"

    randomized_config_path = str(Path(__file__).resolve().parent / "delivery.yaml")

    maniskill_env_id = "DeliveryBase-v1"
    Tools_cls = CycleTool

    situation_init = SituationInit(
        attributes=att
    )

class SimplePreset(BaseTask):
    name = "Simple delivery orders"

    maniskill_env_id = "DeliveryBase-v1"
    Tools_cls = CycleTool

    randomized_config_path = str(Path(__file__).resolve().parent / "delivery.yaml")

    situation_init = SituationInit(
        attributes=att,
        memory={"memory_list":[
            "Packages must be taken from the package storage",
            "Poducts are located in the product storage",
            "After completed the package, you must mark it and return it to the package storage"
        ]}
    )

    PRODUCT_TYPES = ["coca", "icetea", "brets", "donut"]
    MANUFACTURING_ORDERS = ["A47", "B12"]

    def __init__(self, max_products_per_order: int = 1) -> None:
        super().__init__()

        if not 1 <= max_products_per_order <= 4:
            raise ValueError("max_products_per_order must be between 1 and 4")

        table_objects = _sample_table_objects()

        self.assignment = self._sample_assignment(
            table_objects=table_objects,
            max_products_per_order=max_products_per_order,
        )

        self.initialization_parameters = InitializationParameters(
            env_options={"table": table_objects},
            agent_names=["panda"],
        )

        assignment_description = self._assignment_description(self.assignment)

        first_instruction = (
            "Prepare the following two orders using two different "
            f"packages: {assignment_description}"
        )

        self.stages = [
            AtLeastPackageCompletedStage(
                assignments=self.assignment,
                minimum=minimum,
                instruction=(first_instruction if minimum == 1 else "none"),
                flag_answer_to_user=(minimum == len(self.assignment)),
            )
            for minimum in range(1, len(self.assignment) + 1)
        ]


    def _sample_assignment(self, table_objects: list[str], max_products_per_order: int,) -> list[dict]:
        available_objects = table_objects.copy()
        random.shuffle(available_objects)

        assignment = []

        for manufacturing_order in self.MANUFACTURING_ORDERS:
            product_count = random.randint(1, max_products_per_order)

            selected_objects = [available_objects.pop() for _ in range(product_count)]

            selected_types = [object_name.rsplit("_", 1)[0] for object_name in selected_objects]

            assignment.append(
                {
                    "manufacturing_order": manufacturing_order,
                    "products": dict(Counter(selected_types)),
                }
            )

        return assignment

    @staticmethod
    def _assignment_description(assignment: list[dict]) -> str:
        descriptions = []

        for order in assignment:
            products = ", ".join(
                f"{count} {product_type}"
                for product_type, count
                in order["products"].items()
            )

            descriptions.append(
                f"order {order['manufacturing_order']}: {products}"
            )

        return "; ".join(descriptions)

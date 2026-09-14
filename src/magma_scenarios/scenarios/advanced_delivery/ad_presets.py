from copy import deepcopy

from magma_core.simulation.data_structures import (
    SituationInit,
)
from magma_core.simulation.tasks import (
    BaseTask,
    InitializationParameters,
)

from .ad_stages import (
    AtLeastCompletedObjectivesStage,
)

from pathlib import Path
from magma_core.simulation.data_structures import SituationInit
from magma_core.simulation.tasks import BaseTask, InitializationParameters
from .attributes import att, KNOWN_ROBOTS, PRODUCT_TYPES
from .ad_tools import AdvancedDeliveryTools
from .attributes import KNOWN_ROBOTS, att


class ParallelReceptionOneDeliveryPreset(BaseTask):
    name = "Parallel Reception And One Delivery"

    maniskill_env_id = "MultiRobotDelivery-v1"
    Tools_cls = AdvancedDeliveryTools
    randomized_config_path = str(Path(__file__).resolve().parent / "config.yaml")

    def __init__(self) -> None:
        super().__init__()

        products_in_reception = [
            "electronics_0",
            "drinks_0",
        ]

        products_in_storage = [
            "snacks_0",
            "snacks_1",
        ]

        broken_products = [
            "drinks_0",
        ]

        all_products = [
            *products_in_reception,
            *products_in_storage,
        ]

        attributes = deepcopy(att)
        attributes["products"] = all_products

        self.situation_init = SituationInit(
            attributes=attributes,
            memory={
                "memory_list": [
                    (
                        "Incoming products are located in the "
                        "reception area."
                    ),
                    (
                        "Clean returned products must be placed "
                        "in their corresponding storage."
                    ),
                    (
                        "Damaged returned products must be "
                        "discarded."
                    ),
                    (
                        "Priority deliveries must use priority packages "
                        "and priority output bays."
                    ),
                    (
                        "Standard deliveries must use standard packages "
                        "and standard output bays."
                    ),
                ]
            },
        )

        self.initialization_parameters = (
            InitializationParameters(
                env_options={
                    "products_in_reception": (
                        products_in_reception
                    ),
                    "products_in_storage": (
                        products_in_storage
                    ),
                    "broken_products": broken_products,
                },
                agent_names=KNOWN_ROBOTS.copy(),
            )
        )

        delivery_assignment = {
            "order_id": "AD-001",
            "name": "Alice Martin",
            "city": "Paris",
            "service": "priority",
            "products": {
                "snacks": 2,
            },
        }

        physical_assignment = [
            {
                "targets": [
                    "electronics_grid"
                ],
                "products": {
                    "electronics": 1,
                },
            },
            {
                "targets": ["trashcan"],
                "products": {
                    "drinks": 1,
                },
            },
        ]

        fixed_targets_by_object = {
            "drinks_0": "trashcan",
        }

        instruction = (
            "Process the two returned products currently in "
            "the reception area: place the clean product in "
            "its corresponding storage and discard the "
            "damaged product. At the same time, prepare "
            "priority delivery AD-001 for Alice Martin in "
            "Paris with 2 snacks products."
        )

        self.stages = [
            AtLeastCompletedObjectivesStage(
                physical_assignment=(
                    physical_assignment
                ),
                fixed_targets_by_object=(
                    fixed_targets_by_object
                ),
                delivery_assignments=[
                    delivery_assignment
                ],
                reception_product_count=2,
                minimum_completed_goals=minimum,
                instruction=(
                    instruction
                    if minimum == 1
                    else "none"
                ),
                flag_answer=(minimum == 3),
            )
            for minimum in range(1, 4)
        ]

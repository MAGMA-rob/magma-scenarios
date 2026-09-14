import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from magma_core.simulation.data_structures import SituationInit
from magma_core.simulation.state import TaskState
from magma_core.simulation.tasks import InitializationParameters, TaskDefinition
from .ad_tools import AdvancedDeliveryTools
from .attributes import KNOWN_ROBOTS, PRODUCT_TYPES, att
from .requests import ( 
    ProcessReceptionAndDeliveriesRequest,
    AskPendingReturnsCountRequest,
    AskStorageProductCountRequest,
    AskDeliveryDetailsInterruption,
    AskDeliveryServiceCountsInterruption,
    AddReceptionOrDeliveriesInterruption,
    AddDeliveryInProgressRequest,
)

from .requests.common import (
    NEXT_ORDER_NUMBER_PROPERTY,
    OBJECT_CONDITION_RELATION,
    OBJECT_LOCATION_RELATION,
    ORDERS_PROPERTY,
)


PRODUCTS_PER_TYPE = 4
RECEPTION_PRODUCT_COUNT = 5


def _all_products(products_per_type: int = PRODUCTS_PER_TYPE) -> List[str]:
    return [
        f"{product_type}_{index}"
        for product_type in PRODUCT_TYPES
        for index in range(products_per_type)
    ]


def _product_type(object_name: str) -> str:
    product_type, separator, instance_index = (
        object_name.rpartition("_")
    )

    if (
        not separator
        or product_type not in PRODUCT_TYPES
        or not instance_index.isdigit()
    ):
        raise ValueError(
            f"Invalid product instance: {object_name!r}."
        )

    return product_type


def _sample_environment_options(
    products_per_type: int = PRODUCTS_PER_TYPE,
    reception_product_count: int = RECEPTION_PRODUCT_COUNT,
    broken_product_count: Optional[int] = None,
) -> Dict[str, List[str]]:
    if products_per_type < 1:
        raise ValueError("products_per_type must be at least 1.")

    all_products = _all_products(products_per_type)

    if not 1 <= reception_product_count < len(all_products):
        raise ValueError(
            "reception_product_count must leave at least one product in storage."
        )

    if broken_product_count is None:
        broken_product_count = reception_product_count // 2

    if not 0 <= broken_product_count <= reception_product_count:
        raise ValueError(
            "broken_product_count must be between 0 and the reception size."
        )

    products_in_reception = random.sample(
        all_products,
        k=reception_product_count,
    )

    reception_set = set(products_in_reception)

    products_in_storage = [
        object_name
        for object_name in all_products
        if object_name not in reception_set
    ]

    broken_products = random.sample(
        products_in_reception,
        k=broken_product_count,
    )

    return {
        "products_in_reception": products_in_reception,
        "products_in_storage": products_in_storage,
        "broken_products": broken_products,
    }


def _build_object_locations(
    env_options: Dict[str, List[str]],
    all_products: List[str],
) -> Dict[str, str]:
    reception_objects = set(env_options["products_in_reception"])

    return {
        object_name: (
            "reception_grid"
            if object_name in reception_objects
            else f"{_product_type(object_name)}_grid"
        )
        for object_name in all_products
    }


def _build_object_conditions(
    env_options: Dict[str, List[str]],
    all_products: List[str],
) -> Dict[str, str]:
    broken_objects = set(env_options["broken_products"])

    return {
        object_name: (
            "broken"
            if object_name in broken_objects
            else "clean"
        )
        for object_name in all_products
    }


class AdvancedDeliveryDefinition(TaskDefinition):
    name = "Advanced Delivery Definition"

    maniskill_env_id = "MultiRobotDelivery-v1"
    Tools_cls = AdvancedDeliveryTools

    randomized_config_path = str(
        Path(__file__).resolve().parent / "config.yaml"
    )

    active_requests = [
        ProcessReceptionAndDeliveriesRequest(),
        AskStorageProductCountRequest(),
        AskPendingReturnsCountRequest(),
        AskDeliveryDetailsInterruption(),
        AskDeliveryServiceCountsInterruption(),
        AddReceptionOrDeliveriesInterruption(),
        AddDeliveryInProgressRequest(),
    ]

    def __init__(
        self,
        products_per_type: int = PRODUCTS_PER_TYPE,
        reception_product_count: int = RECEPTION_PRODUCT_COUNT,
        broken_product_count: Optional[int] = None,
    ) -> None:
        env_options = _sample_environment_options(
            products_per_type=products_per_type,
            reception_product_count=reception_product_count,
            broken_product_count=broken_product_count,
        )
        all_products = _all_products(products_per_type)

        attributes = deepcopy(att)

        starting_state = TaskState()

        starting_state.attributes = deepcopy(attributes)

        starting_state.relations[OBJECT_LOCATION_RELATION] = (
            _build_object_locations(env_options, all_products)
        )

        starting_state.relations[OBJECT_CONDITION_RELATION] = (
            _build_object_conditions(env_options, all_products)
        )

        starting_state.properties[NEXT_ORDER_NUMBER_PROPERTY] = 1

        starting_state.properties[ORDERS_PROPERTY] = {}

        situation_init = SituationInit(
            attributes=deepcopy(attributes),
            all_task_attributes=deepcopy(attributes),
            memory={
                "memory_list": [
                    (
                        "Incoming products are unloaded in the "
                        "reception area."
                    ),
                    (
                        "Damaged products must be discarded, "
                        "while clean unused products must be "
                        "returned to their corresponding storage."
                    ),
                    (
                        "Priority deliveries must use priority packages "
                        "and priority output bays."
                    ),
                    (
                        "Every completed package must receive the "
                        "shipping label associated with its "
                        "delivery."
                    ),
                ]
            },
        )

        initialization_parameters = (
            InitializationParameters(
                env_options=deepcopy(env_options),
                agent_names=KNOWN_ROBOTS.copy(),
            )
        )

        super().__init__(
            name=self.name,
            situation_init=situation_init,
            starting_state=starting_state,
            initialization_parameters=(
                initialization_parameters
            ),
        )


class SimpleReceptionDeliveryDefinition(AdvancedDeliveryDefinition):
    name = "Simple Reception And Delivery Definition"

    def __init__(self) -> None:
        super().__init__(
            products_per_type=2,
            reception_product_count=2,
            broken_product_count=1,
        )
        self.active_requests = [
            ProcessReceptionAndDeliveriesRequest(
                max_deliveries=1,
                max_products_per_delivery=2,
                reception_size=2,
                broken_count=1,
            ),
            AskStorageProductCountRequest(),
            AskPendingReturnsCountRequest(),
        ]

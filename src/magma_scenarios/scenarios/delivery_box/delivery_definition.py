from copy import deepcopy
from pathlib import Path
from magma_core.simulation.data_structures import SituationInit
from magma_core.simulation.state import TaskState
from magma_core.simulation.tasks import InitializationParameters, TaskDefinition

from .attributes import _sample_table_objects, att
from .delivery_tools import CycleTool
from .requests.orders import SendOrdersRequest
from .requests.interruptions import (
    OrderWithHighPriority,
    OrderWithLowPriorityInterrupt,
)
from .requests.promotions import (
    AddThirdProductRequest,
    AlwaysAddGiftRequest,
    ForgetRuleRequest,
)
from .requests.constants import REQUESTS_SINCE_PROMOTION_KEY
from .rule_renderer import DeliveryRuleRenderer


class MultiOrderDeliveryDefinition(TaskDefinition):
    """
    Continuous single-robot delivery scenario.

    The robot receives pairs of orders, prepares each order in a
    different package and marks each package with its order reference.

    Promotion requests can modify future orders. An interruption can
    also request that a previously created order be remade urgently.
    """

    maniskill_env_id = "DeliveryBase-v1"
    Tools_cls = CycleTool
    RuleRenderer_cls = DeliveryRuleRenderer

    active_requests = [
        # Creates two orders using products currently available.
        SendOrdersRequest(number_of_orders=2, max_products_per_order=3),

        # Add an order announced while the initial orders are being completed.
        OrderWithLowPriorityInterrupt(
            number_of_orders=2,
            max_products_per_order=3,
        ),

        # Interrupt one in-progress order with an urgent order.
        OrderWithHighPriority(max_products_per_order=3),

        # Permanent promotion: always add one gift product.
        AlwaysAddGiftRequest(),

        # Permanent promotion: add a third product when an order
        AddThirdProductRequest(),

        # Remove the active promotion with an increasing sampling weight.
        ForgetRuleRequest(),
    ]

    def __init__(self) -> None:
        table_objects = _sample_table_objects()

        attributes = deepcopy(att)

        starting_state = TaskState()
        starting_state.attributes = deepcopy(attributes)
        starting_state.relations["order_history"] = {}
        starting_state.properties["table_objects"] = deepcopy(table_objects)
        starting_state.properties["next_order_number"] = 1
        starting_state.properties[REQUESTS_SINCE_PROMOTION_KEY] = 0

        super().__init__(
            name="Multi-order delivery with promotions and interruptions",
            situation_init=SituationInit(
                attributes=attributes,
            ),
            starting_state=starting_state,
            initialization_parameters=InitializationParameters(
                env_options={"table": table_objects,},
                agent_names=["panda"],
            ),
            randomized_config_path=str(
                Path(__file__).resolve().parent / "delivery.yaml"
            ),
        )

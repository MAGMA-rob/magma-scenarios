from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, TypedDict

from magma_core.simulation.state import TaskState
from magma_core.utils.text_utils import join_with_and

from ..attributes import PRODUCT_TYPES


PRIORITY_SERVICE = "priority"
STANDARD_SERVICE = "standard"
OBJECT_LOCATION_RELATION = "object_location"
OBJECT_CONDITION_RELATION = "object_condition"
ORDERS_PROPERTY = "orders"
NEXT_ORDER_NUMBER_PROPERTY = "next_order_number"

DeliveryService = Literal["priority", "standard"]


class DeliveryAssignment(TypedDict):
    order_id: str
    name: str
    city: str
    service: DeliveryService
    products: Dict[str, int]


@dataclass(frozen=True)
class DeliveryOrder:
    order_id: str
    name: str
    city: str
    service: DeliveryService
    products: Tuple[Tuple[str, int], ...]

    @classmethod
    def from_assignment(cls, assignment: DeliveryAssignment) -> "DeliveryOrder":
        return cls(
            assignment["order_id"],
            assignment["name"],
            assignment["city"],
            assignment["service"],
            tuple(assignment["products"].items()),
        )

    def to_assignment(self) -> DeliveryAssignment:
        return {
            "order_id": self.order_id,
            "name": self.name,
            "city": self.city,
            "service": self.service,
            "products": dict(self.products),
        }


CUSTOMERS = [
    {
        "name": "Alice Martin",
        "city": "Paris",
    },
    {
        "name": "Karim Benali",
        "city": "Lyon",
    },
    {
        "name": "Emma Laurent",
        "city": "Marseille",
    },
    {
        "name": "Nadia Petit",
        "city": "Toulouse",
    },
    {
        "name": "Lucas Moreau",
        "city": "Nantes",
    },
    {
        "name": "Sofia Bernard",
        "city": "Bordeaux",
    },
]


def build_order_id(order_number: int) -> str:
    if order_number < 1:
        raise ValueError(
            "order_number must be at least 1."
        )

    return f"AD-{order_number:03d}"


def next_order_id(state: TaskState, offset: int = 0) -> str:
    if offset < 0:
        raise ValueError("offset must be non-negative.")

    next_number = state.properties.get(NEXT_ORDER_NUMBER_PROPERTY, 1)

    return build_order_id(next_number + offset)


def validate_products(products: Dict[str, int]) -> Dict[str, int]:
    if not isinstance(products, dict) or not products:
        raise ValueError("products must be a non-empty dictionary.")

    normalized = {}

    for product_type, quantity in products.items():
        if product_type not in PRODUCT_TYPES:
            raise ValueError(f"Unknown product type: {product_type}.")

        if not isinstance(quantity, int) or quantity < 1:
            raise ValueError(f"Quantity for {product_type} must be >= 1.")

        normalized[product_type] = quantity

    return normalized


def validate_assignment(assignment: DeliveryAssignment) -> DeliveryAssignment:
    order_id = assignment.get("order_id")
    name = assignment.get("name")
    city = assignment.get("city")
    service = assignment.get("service")
    products = assignment.get("products")

    if not isinstance(order_id, str) or not order_id:
        raise ValueError(
            "A delivery requires an order_id."
        )

    if not isinstance(name, str) or not name:
        raise ValueError(
            "A delivery requires a recipient name."
        )

    if not isinstance(city, str) or not city:
        raise ValueError(
            "A delivery requires a destination city."
        )

    if service not in {
        PRIORITY_SERVICE,
        STANDARD_SERVICE,
    }:
        raise ValueError(
            "Delivery service must be priority or standard."
        )

    return {
        "order_id": order_id,
        "name": name,
        "city": city,
        "service": service,
        "products": validate_products(products),
    }


def products_description(products: Dict[str, int]) -> str:
    descriptions = [
        (
            f"{quantity} {product_type} product"
            if quantity == 1
            else f"{quantity} {product_type} products"
        )
        for product_type, quantity in products.items()
    ]

    return join_with_and(descriptions)


def delivery_description(assignment: DeliveryAssignment) -> str:
    return (
        f"delivery {assignment['order_id']} for "
        f"{assignment['name']} in {assignment['city']}, "
        f"using {assignment['service']} service, containing "
        f"{products_description(assignment['products'])}"
    )


def deliveries_description(assignments: List[DeliveryAssignment]) -> str:
    return join_with_and(
        [
            delivery_description(assignment)
            for assignment in assignments
        ]
    )


def save_assignments(state: TaskState, assignments: List[DeliveryAssignment]) -> TaskState:
    """
    Save generated deliveries after a request and reserve the
    following order identifiers.
    """

    history = deepcopy(
        state.properties.get(
            ORDERS_PROPERTY,
            {},
        )
    )

    for assignment in assignments:
        normalized = validate_assignment(assignment)

        history[normalized["order_id"]] = deepcopy(
            normalized
        )

    state.properties[ORDERS_PROPERTY] = history

    state.properties[NEXT_ORDER_NUMBER_PROPERTY] = (
        state.properties.get(
            NEXT_ORDER_NUMBER_PROPERTY,
            1,
        )
        + len(assignments)
    )

    return state

from typing import Dict, List, Sequence

from magma_core.utils.text_utils import join_with_and


def product_sequence(assignment: Dict) -> List[str]:
    products: List[str] = []
    for product_type, quantity in assignment["products"].items():
        products.extend([product_type] * quantity)
    return products


def products_description(assignment: Dict) -> str:
    return join_with_and([
        f"{quantity} {'unit' if quantity == 1 else 'units'} of {product_type}"
        for product_type, quantity in assignment["products"].items()
    ])


def order_description(assignment: Dict) -> str:
    return (
        f"order {assignment['manufacturing_order']} with "
        f"{products_description(assignment)}"
    )


def orders_description(assignments: Sequence[Dict]) -> str:
    return join_with_and([order_description(assignment) for assignment in assignments])


def orders_ready_message(assignments: Sequence[Dict]) -> str:
    label = "order is" if len(assignments) == 1 else "orders are"
    return f"The following {label} ready: {orders_description(assignments)}."

# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Dict, List, Tuple
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from ..delivery_stages import AtLeastPackageCompletedStage
from .constants import GIFT_PROMOTION, THIRD_PRODUCT_PROMOTION
from .promotions import active_promotion, increment_promotion_age
from .common import (
    order_description,
    orders_description,
    orders_ready_message,
    product_sequence,
    products_description,
)
from magma_core.simulation.stage import BaseTaskStage

class SendOrdersRequest(BaseRequest):

    @dataclass(frozen=True)
    class Parameters:
        assignments: Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...]
        instruction: str

        def order_dicts(self) -> List[Dict]:
            return [
                {
                    "manufacturing_order": order_number,
                    "products": dict(products),
                }
                for order_number, products in self.assignments
            ]

    def __init__(self, number_of_orders: int = 2, max_products_per_order: int = 3) -> None:
        super().__init__()

        if number_of_orders < 1:
            raise ValueError(
                "number_of_orders must be at least 1."
            )

        if max_products_per_order < 1:
            raise ValueError(
                "max_products_per_order must be at least 1."
            )

        self.number_of_orders = number_of_orders
        self.max_products_per_order = max_products_per_order

    def sampling_weight(self, state: TaskState) -> float:
        return float(self._maximum_feasible_order_count(state) >= 1)

    def sample_parameters(self, state: TaskState) -> Parameters:
        maximum_orders = self._maximum_feasible_order_count(state)
        if maximum_orders < 1:
            raise RuntimeError("SendOrdersRequest requires products on the table.")

        selected_number_of_orders = random.randint(1, maximum_orders)
        assignments, instruction_assignments = self._create_assignments(
            state=state,
            number_of_orders=selected_number_of_orders,
        )
        paired_assignments = sorted(
            zip(assignments, instruction_assignments),
            key=lambda pair: sum(pair[0]["products"].values()),
        )
        assignments = [pair[0] for pair in paired_assignments]
        instruction_assignments = [pair[1] for pair in paired_assignments]

        assignment_description = self._assignment_description(instruction_assignments)
        instruction = (
            f"Prepare the following {selected_number_of_orders} order(s) "
            f"using different packages: {assignment_description}"
        )
        return self.Parameters(
            tuple(
                (assignment["manufacturing_order"], tuple(assignment["products"].items()))
                for assignment in assignments
            ),
            instruction,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: Parameters,
    ) -> List[BaseTaskStage]:
        return self._generic_stages(
            assignments=parameters.order_dicts(),
            first_instruction=parameters.instruction,
        )


    def _maximum_feasible_order_count(self, state: TaskState) -> int:
        table_objects = state.properties.get("table_objects", [])
        maximum = min(self.number_of_orders, len(table_objects))

        promotion = active_promotion(state)
        if promotion is not None and promotion["kind"] == GIFT_PROMOTION:
            available_counts = self._available_counts(table_objects)
            maximum = min(maximum, available_counts.get(promotion["product"], 0))
        return maximum

    def _create_assignments(
        self,
        state: TaskState,
        number_of_orders: int,
        first_order_minimum: int = 1,
    ) -> tuple[List[Dict], List[Dict]]:
        promotion = active_promotion(state)
        return self._build_assignment(
            available_counts=self._available_counts(
                state.properties.get("table_objects", [])
            ),
            first_order_number=state.properties.get("next_order_number", 1),
            number_of_orders=number_of_orders,
            first_order_minimum=first_order_minimum,
            gift_product=(
                promotion["product"]
                if promotion is not None and promotion["kind"] == GIFT_PROMOTION
                else None
            ),
            add_third_product=(
                promotion is not None
                and promotion["kind"] == THIRD_PRODUCT_PROMOTION
            ),
        )

    @staticmethod
    def _generic_stages(
        assignments: List[Dict],
        first_instruction: str,
        additional_instructions: Dict[int, str] | None = None,
        flag_answer_on_last: bool = True,
        wait_after_stage: int | None = None,
        linked_to_prev_stages: set[int] | None = None,
    ) -> List[BaseTaskStage]:
        instructions = additional_instructions or {}
        linked_stages = linked_to_prev_stages or set()
        stages = [
            AtLeastPackageCompletedStage(
                assignments=assignments,
                minimum=minimum,
                instruction=(
                    first_instruction
                    if minimum == 1
                    else instructions.get(minimum, "none")
                ),
                flag_answer_to_user=(
                    flag_answer_on_last and minimum == len(assignments)
                ),
                move_to_packages_before_take=minimum == 1,
                wait_after_completion=minimum == wait_after_stage,
                linked_to_prev=minimum in linked_stages,
            )
            for minimum in range(1, len(assignments) + 1)
        ]
        return stages

    @staticmethod
    def _product_sequence(assignment: Dict) -> List[str]:
        return product_sequence(assignment)

    @staticmethod
    def _products_description(assignment: Dict) -> str:
        return products_description(assignment)

    @classmethod
    def _order_description(cls, assignment: Dict) -> str:
        return order_description(assignment)

    @classmethod
    def _orders_description(cls, assignments: List[Dict]) -> str:
        return orders_description(assignments)

    @classmethod
    def _orders_ready_message(cls, assignments: List[Dict]) -> str:
        return orders_ready_message(assignments)

    def apply_request(
        self,
        state: TaskState,
        parameters: Parameters,
    ) -> TaskState:
        """Save generated orders and age the active promotion, if any."""
        history = deepcopy(state.relations.get("order_history", {}))

        assignments = parameters.order_dicts()
        for order in assignments:
            order_id = order["manufacturing_order"]

            history[order_id] = {"products": deepcopy(order["products"])}

        state.relations["order_history"] = history

        state.properties["next_order_number"] = (
            state.properties.get("next_order_number", 1) + len(assignments)
        )

        increment_promotion_age(state)

        return state

        
    def _build_assignment(
        self,
        available_counts: Counter,
        first_order_number: int,
        number_of_orders: int,
        first_order_minimum: int,
        gift_product: str | None,
        add_third_product: bool,
    ) -> tuple[List[Dict], List[Dict]]:
        remaining = available_counts.copy()

        normal_assignments = [
            {
                "manufacturing_order": (f"{first_order_number + index:03d}"),
                "products": Counter(),
            }
            for index in range(number_of_orders)
        ]
        if gift_product is not None:
            self._build_orders_with_gift(
                assignments=normal_assignments,
                remaining=remaining,
                gift_product=gift_product,
            )

        elif add_third_product:
            self._build_orders_with_third_product(
                assignments=normal_assignments,
                remaining=remaining,
                first_order_minimum=first_order_minimum,
            )

        else:
            self._build_normal_orders(
                assignments=normal_assignments,
                remaining=remaining,
                first_order_minimum=first_order_minimum,
            )

        self._assert_assignments_fit_available(
            assignments=normal_assignments,
            available_counts=available_counts,
        )
        instruction_assignments = deepcopy(normal_assignments)

        if gift_product is not None:
            for order in instruction_assignments:
                order["products"][gift_product] -= 1

                if order["products"][gift_product] == 0:
                    del order["products"][gift_product]

        elif add_third_product:
            for order in instruction_assignments:
                for product_type, quantity in list(order["products"].items()):
                    if quantity == 3:
                        order["products"][product_type] = 2

        normal_assignments = [
            {
                "manufacturing_order": order["manufacturing_order"],
                "products": dict(order["products"]),
            }
            for order in normal_assignments
        ]
        instruction_assignments = [
            {
                "manufacturing_order": order["manufacturing_order"],
                "products": dict(order["products"]),
            }
            for order in instruction_assignments
        ]

        return normal_assignments, instruction_assignments

    @staticmethod
    def _assert_assignments_fit_available(
        assignments: List[Dict],
        available_counts: Counter,
    ) -> None:
        requested_counts = Counter()
        for assignment in assignments:
            requested_counts.update(assignment["products"])

        unavailable = {
            product_type: {
                "requested": requested_count,
                "available": available_counts.get(product_type, 0),
            }
            for product_type, requested_count in requested_counts.items()
            if requested_count > available_counts.get(product_type, 0)
        }
        if unavailable:
            raise RuntimeError(
                "Generated orders exceed the available product stock: "
                f"{unavailable}"
            )

    def _build_orders_with_gift(self, assignments: List[Dict], remaining: Counter, gift_product: str | None) -> None:
        if gift_product is None:
            raise RuntimeError("The gift promotion is active but no gift product was selected.")

        number_of_orders = len(assignments)
        if remaining.get(gift_product, 0) < number_of_orders:
            raise RuntimeError(f"Not enough {gift_product} products for the gift promotion.")

        remaining[gift_product] -= number_of_orders

        for order in assignments:
            
            available_products = [
                product_type for product_type, quantity in remaining.items() 
                if ( quantity > 0 and product_type != gift_product)
            ]

            if not available_products:
                raise RuntimeError("No product remains available to build an order besides the gift.")

            selected_product = random.choice(available_products)

            order["products"][selected_product] += 1
            remaining[selected_product] -= 1

            order["products"][gift_product] += 1

    def _build_orders_with_third_product(
        self,
        assignments: List[Dict],
        remaining: Counter,
        first_order_minimum: int,
    ) -> None:
        for index, order in enumerate(assignments):
            remaining_orders = len(assignments) - index - 1
            candidates_with_three = [
                product_type
                for product_type, quantity in remaining.items()
                if quantity >= 3 and sum(remaining.values()) >= 3 + remaining_orders
            ]

            if candidates_with_three:
                selected_product = random.choice(candidates_with_three)

                order["products"][selected_product] += 3
                remaining[selected_product] -= 3

                continue

            minimum = first_order_minimum if index == 0 else 1
            available_products = [
                product_type
                for product_type, quantity in remaining.items()
                if quantity > 0
            ]

            if len(available_products) < minimum:
                raise RuntimeError(
                    "Not enough distinct products to build the interrupted order "
                    "without violating the third-product promotion."
                )

            for selected_product in random.sample(available_products, k=minimum):
                order["products"][selected_product] += 1
                remaining[selected_product] -= 1

    def _build_normal_orders(
        self,
        assignments: List[Dict],
        remaining: Counter,
        first_order_minimum: int,
    ) -> None:
        for index, order in enumerate(assignments):
            remaining_orders = (len(assignments) - index - 1)

            available_count = sum(remaining.values())

            maximum_products = min(self.max_products_per_order, available_count - remaining_orders)

            minimum_products = first_order_minimum if index == 0 else 1
            if maximum_products < minimum_products:
                raise RuntimeError("Not enough products to build all orders.")

            product_count = random.randint(minimum_products, maximum_products)

            for _ in range(product_count):
                available_products = [product_type for product_type, quantity in remaining.items() if quantity > 0]

                selected_product = random.choice(available_products)

                order["products"][selected_product] += 1
                remaining[selected_product] -= 1

    @staticmethod
    def _available_counts(table_objects: List[str]) -> Counter:
        """
        Convert objects such as coca_1 and coca_2 into
        product counts such as {"coca": 2}.
        """
        return Counter(object_name.rsplit("_", 1)[0] for object_name in table_objects)


    @staticmethod
    def _assignment_description(assignment: List[Dict]) -> str:
        descriptions = []

        for order in assignment:
            product_description = ", ".join(f"{quantity} {product_type}"
                for product_type, quantity in order["products"].items())

            descriptions.append(
                f"order {order['manufacturing_order']}: "
                f"{product_description}"
            )

        return "; ".join(descriptions)

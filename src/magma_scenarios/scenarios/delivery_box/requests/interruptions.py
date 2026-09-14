from dataclasses import dataclass
import random
from typing import List, Tuple

from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState

from ..attributes import PACKAGE_NAMES
from ..delivery_stages import (
    AtLeastPackageCompletedStage,
    AtLeastProductsInHeldPackageStage,
    EmptyPackageTakenStage,
    ValidatePackageStage,
)
from .constants import GIFT_PROMOTION
from .orders import SendOrdersRequest
from .promotions import active_promotion


@dataclass(frozen=True)
class LowPriorityParameters:
    orders: SendOrdersRequest.Parameters
    initial_order_count: int
    announcement: str
    announcement_stage: int


class OrderWithLowPriorityInterrupt(SendOrdersRequest):
    """Add an order to complete after an initial batch of orders."""

    def __init__(
        self,
        number_of_orders: int = 2,
        max_products_per_order: int = 3,
    ) -> None:
        if number_of_orders < 2:
            raise ValueError("number_of_orders must be at least 2")
        if number_of_orders >= len(PACKAGE_NAMES):
            raise ValueError(
                "number_of_orders must leave one package for the low-priority order"
            )
        super().__init__(number_of_orders, max_products_per_order)

    def sampling_weight(self, state: TaskState) -> float:
        return float(self._maximum_initial_order_count(state) >= 2)

    def sample_parameters(self, state: TaskState) -> LowPriorityParameters:
        maximum_initial_orders = self._maximum_initial_order_count(state)
        if maximum_initial_orders < 2:
            raise RuntimeError(
                "OrderWithLowPriorityInterrupt requires enough products for at "
                "least two initial orders and one low-priority order."
            )
        initial_count = random.randint(2, maximum_initial_orders)
        assignments, instruction_assignments = self._create_assignments(
            state=state,
            number_of_orders=initial_count + 1,
        )
        paired_initial = sorted(
            zip(assignments[:initial_count], instruction_assignments[:initial_count]),
            key=lambda pair: sum(pair[0]["products"].values()),
        )
        assignments = [pair[0] for pair in paired_initial] + [assignments[-1]]
        instruction_assignments = [
            pair[1] for pair in paired_initial
        ] + [instruction_assignments[-1]]
        initial_description = self._assignment_description(
            instruction_assignments[:initial_count]
        )
        first_instruction = (
            f"Prepare the following {initial_count} order(s) using different "
            f"packages: {initial_description}"
        )
        low_priority_description = self._assignment_description(
            [instruction_assignments[-1]]
        )
        announcement = random.choice([
            (
                f"A new order has arrived: {low_priority_description}. Complete "
                "it after the current orders."
            ),
            (
                "New recipe to prepare after the current orders: "
                f"{low_priority_description}."
            ),
            (
                f"Complete {low_priority_description} once you have finished "
                "the current orders."
            ),
        ])
        return LowPriorityParameters(
            SendOrdersRequest.Parameters(
                tuple(
                    (
                        assignment["manufacturing_order"],
                        tuple(assignment["products"].items()),
                    )
                    for assignment in assignments
                ),
                first_instruction,
            ),
            initial_count,
            announcement,
            random.randint(2, initial_count),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: LowPriorityParameters,
    ) -> List[BaseTaskStage]:
        assignments = parameters.orders.order_dicts()
        stages = self._generic_stages(
            assignments=assignments[:parameters.initial_order_count],
            first_instruction=parameters.orders.instruction,
            additional_instructions={
                parameters.announcement_stage: parameters.announcement
            },
            flag_answer_on_last=False,
            wait_after_stage=parameters.announcement_stage - 1,
            linked_to_prev_stages={parameters.announcement_stage},
        )
        stages.append(AtLeastPackageCompletedStage(
            assignments=assignments,
            minimum=len(assignments),
            instruction="none",
            flag_answer_to_user=True,
        ))
        return stages


    def apply_request(
        self,
        state: TaskState,
        parameters: LowPriorityParameters,
    ) -> TaskState:
        return super().apply_request(state, parameters.orders)

    def _maximum_initial_order_count(self, state: TaskState) -> int:
        table_objects = state.properties.get("table_objects", [])
        maximum = min(
            self.number_of_orders,
            len(table_objects) - 1,
            len(PACKAGE_NAMES) - 1,
        )
        promotion = active_promotion(state)
        if promotion is not None and promotion["kind"] == GIFT_PROMOTION:
            available_counts = self._available_counts(table_objects)
            maximum = min(
                maximum,
                available_counts.get(promotion["product"], 0) - 1,
            )
        return maximum


@dataclass(frozen=True)
class HighPriorityParameters:
    orders: SendOrdersRequest.Parameters
    initial_instruction: str
    priority_instruction: str
    interrupt_after: int
    objects_in_interrupted_package: Tuple[str, ...]


class OrderWithHighPriority(SendOrdersRequest):
    """Interrupt one order with a second order that must be completed first."""

    def __init__(self, max_products_per_order: int = 3) -> None:
        super().__init__(2, max_products_per_order)

    def sampling_weight(self, state: TaskState) -> float:
        promotion = active_promotion(state)
        minimum_stock = 4 if promotion is not None else 3
        return float(
            self._maximum_feasible_order_count(state) >= 2
            and len(state.properties.get("table_objects", [])) >= minimum_stock
        )

    def sample_parameters(self, state: TaskState) -> HighPriorityParameters:
        if self.sampling_weight(state) == 0:
            raise RuntimeError(
                "OrderWithHighPriority requires enough products for two orders."
            )
        assignments, instruction_assignments = self._create_assignments(
            state=state,
            number_of_orders=2,
            first_order_minimum=2,
        )
        initial_order, _ = assignments
        initial_instruction = (
            f"Prepare {self._assignment_description([instruction_assignments[0]])}."
        )
        priority_instruction = (
            "Urgent interruption: a new order has arrived: "
            f"{self._assignment_description([instruction_assignments[1]])}. "
            "Stop the current order, complete this new order first, then resume "
            "the interrupted order."
        )
        initial_products = self._product_sequence(initial_order)
        interrupt_after = random.randint(1, len(initial_products) - 1)
        available_by_type = {
            product_type: [
                object_name
                for object_name in state.properties.get("table_objects", [])
                if object_name.rsplit("_", 1)[0] == product_type
            ]
            for product_type in initial_order["products"]
        }
        objects_in_package = []
        for product_type in initial_products[:interrupt_after]:
            objects_in_package.append(available_by_type[product_type].pop(0))
        return HighPriorityParameters(
            SendOrdersRequest.Parameters(
                tuple(
                    (
                        assignment["manufacturing_order"],
                        tuple(assignment["products"].items()),
                    )
                    for assignment in assignments
                ),
                initial_instruction,
            ),
            initial_instruction,
            priority_instruction,
            interrupt_after,
            tuple(objects_in_package),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: HighPriorityParameters,
    ) -> List[BaseTaskStage]:
        initial_order, priority_order = parameters.orders.order_dicts()
        stages: List[BaseTaskStage] = [
            EmptyPackageTakenStage(instruction=parameters.initial_instruction)
        ]
        initial_products = self._product_sequence(initial_order)
        for product_index, _ in enumerate(
            initial_products[:parameters.interrupt_after]
        ):
            stages.append(AtLeastProductsInHeldPackageStage(
                products=initial_order["products"],
                minimum=product_index + 1,
                inspect_empty_package=product_index == 0,
            ))
        stages.append(EmptyPackageTakenStage(
            instruction=parameters.priority_instruction,
            replace_held_package=True,
            linked_to_prev=True,
        ))
        for product_index, _ in enumerate(self._product_sequence(priority_order)):
            stages.append(AtLeastProductsInHeldPackageStage(
                products=priority_order["products"],
                minimum=product_index + 1,
                inspect_empty_package=product_index == 0,
            ))
        stages.append(ValidatePackageStage(priority_order))
        for missing_index, _ in enumerate(
            initial_products[parameters.interrupt_after:]
        ):
            stages.append(AtLeastProductsInHeldPackageStage(
                products=initial_order["products"],
                minimum=parameters.interrupt_after + missing_index + 1,
                resume_interrupted_package=missing_index == 0,
            ))
        stages.append(ValidatePackageStage(
            initial_order,
            flag_answer_to_user=True,
        ))
        return stages


    def apply_request(
        self,
        state: TaskState,
        parameters: HighPriorityParameters,
    ) -> TaskState:
        return super().apply_request(state, parameters.orders)

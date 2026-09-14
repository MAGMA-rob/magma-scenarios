import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from ..ad_stages import AtLeastCompletedObjectivesStage
from ..attributes import PRODUCT_TYPES
from ..environment_transitions import RECEPTION_OFFSETS, ResetAdvancedDeliveryTransition
from ..utils import product_type_from_name
from .common import (
    CUSTOMERS,
    NEXT_ORDER_NUMBER_PROPERTY,
    OBJECT_CONDITION_RELATION,
    OBJECT_LOCATION_RELATION,
    PRIORITY_SERVICE,
    STANDARD_SERVICE,
    DeliveryAssignment,
    DeliveryOrder,
    build_order_id,
    deliveries_description,
    save_assignments,
    DeliveryService,
)


AVAILABLE_SERVICES: List[DeliveryService] = [PRIORITY_SERVICE, STANDARD_SERVICE]
PACKAGES_PER_SERVICE = 2
MAX_CONCURRENT_DELIVERIES = PACKAGES_PER_SERVICE * len(AVAILABLE_SERVICES)

CLEAN_CONDITION = "clean"
BROKEN_CONDITION = "broken"

RECEPTION_LOCATION = "reception_grid"
TRASH_LOCATION = "trashcan"

NEW_RECEPTION_SIZE = 5
NEW_BROKEN_COUNT = 2


@dataclass(frozen=True)
class ProcessingParameters:
    assignments: Tuple[DeliveryOrder, ...]
    current_reception_objects: Tuple[str, ...]
    next_reception_objects: Tuple[str, ...]
    next_broken_objects: Tuple[str, ...]
    reception_offsets: Tuple[Tuple[float, float], ...]

    def assignment_dicts(self) -> List[DeliveryAssignment]:
        return [order.to_assignment() for order in self.assignments]

    @property
    def instruction(self) -> str:
        assignments = self.assignment_dicts()
        return (
            "There is a client return in the reception. At the same time, "
            f"prepare {len(assignments)} "
            f"{'delivery' if len(assignments) == 1 else 'deliveries'}: "
            f"{deliveries_description(assignments)}."
        )


def _product_grid(object_name: str) -> str:
    product_type = product_type_from_name(object_name)

    if product_type is None:
        raise ValueError(f"Invalid product instance: {object_name!r}.")

    return f"{product_type}_grid"


def _all_products(state: TaskState) -> List[str]:
    object_locations = state.relations.get(OBJECT_LOCATION_RELATION, {})

    if not object_locations:
        raise RuntimeError("No product locations are registered.")

    return list(object_locations)

def _required_product_type(object_name: str,) -> str:
    product_type = product_type_from_name(
        object_name
    )

    if product_type is None:
        raise ValueError(f"Invalid product instance: {object_name!r}.")

    return product_type


def processing_objective_count(
    reception_assignment: List[Dict],
    fixed_targets: Dict[str, str],
    delivery_count: int,
) -> int:
    has_clean_returns = any(
        TRASH_LOCATION not in assignment.get("targets", [])
        for assignment in reception_assignment
    )
    return (
        int(has_clean_returns)
        + int(bool(fixed_targets))
        + delivery_count
    )


class ProcessReceptionAndDeliveriesRequest(BaseRequest[ProcessingParameters]):
    def __init__(
        self,
        max_deliveries: int = 2,
        max_products_per_delivery: int = 3,
        reception_size: int = NEW_RECEPTION_SIZE,
        broken_count: int = NEW_BROKEN_COUNT,
        weight: float = 5.0,
    ) -> None:
        super().__init__()

        if max_deliveries < 1:
            raise ValueError("max_deliveries must be at least 1.")

        if max_deliveries > MAX_CONCURRENT_DELIVERIES:
            raise ValueError(
                "max_deliveries cannot exceed the number of available packages."
            )

        if max_products_per_delivery < 1:
            raise ValueError("max_products_per_delivery must be at least 1.")

        if reception_size < 2:
            raise ValueError("reception_size must be at least 2.")

        if broken_count < 1:
            raise ValueError("broken_count must be at least 1.")

        if broken_count > reception_size:
            raise ValueError("broken_count cannot exceed reception_size.")

        if weight < 0:
            raise ValueError(
                "weight must be non-negative."
            )

        self.max_deliveries = max_deliveries
        self.max_products_per_delivery = (max_products_per_delivery)
        self.reception_size = reception_size
        self.broken_count = broken_count
        self.weight = weight


    def _reception_objects(
        self,
        state: TaskState,
    ) -> List[str]:
        locations = state.relations.get(OBJECT_LOCATION_RELATION, {})

        return [
            object_name
            for object_name in _all_products(state)
            if locations.get(object_name)
            == RECEPTION_LOCATION
        ]

    def _available_delivery_products(self, state: TaskState) -> List[str]:
        locations = state.relations.get(OBJECT_LOCATION_RELATION, {})

        conditions = state.relations.get(OBJECT_CONDITION_RELATION, {})

        return [
            object_name
            for object_name in _all_products(state)
            if (
                conditions.get(object_name)
                == CLEAN_CONDITION
                and locations.get(object_name)
                == _product_grid(object_name)
            )
        ]

    def sampling_weight(self, state: TaskState) -> float:
        if not self._reception_objects(state):
            return 0

        if not self._available_delivery_products(
            state
        ):
            return 0

        return self.weight

    def _build_deliveries(
        self,
        state: TaskState,
        available_products: List[str],
        delivery_count: int,
    ) -> List[DeliveryAssignment]:
        remaining_products = (available_products.copy())

        random.shuffle(remaining_products)

        customers = random.sample(CUSTOMERS, k=delivery_count)

        services = random.choices(AVAILABLE_SERVICES, k=delivery_count)
        while max(Counter(services).values()) > PACKAGES_PER_SERVICE:
            services = random.choices(AVAILABLE_SERVICES, k=delivery_count)
        next_order_number = state.properties.get(NEXT_ORDER_NUMBER_PROPERTY, 1)

        assignments: List[DeliveryAssignment] = []

        for index in range(delivery_count):
            remaining_deliveries = (delivery_count - index - 1)

            maximum_product_count = min(
                self.max_products_per_delivery,
                len(remaining_products)
                - remaining_deliveries,
            )

            product_count = random.randint(1, maximum_product_count)

            selected_instances = [
                remaining_products.pop()
                for _ in range(product_count)
            ]

            products: Dict[str, int] = dict(
                Counter(
                    _required_product_type(object_name)
                    for object_name in selected_instances
                )
            )

            customer = customers[index]

            assignments.append(
                {
                    "order_id": build_order_id(next_order_number + index),
                    "name": customer["name"],
                    "city": customer["city"],
                    "service": services[index],
                    "products": products,
                }
            )

        return assignments

    def _build_reception_assignment(
        self,
        state: TaskState,
        assignments: List[DeliveryAssignment],
    ) -> tuple[List[Dict], Dict[str, str],]:
        all_products = _all_products(state)

        conditions = state.relations.get(OBJECT_CONDITION_RELATION,{})

        reception_objects = (self._reception_objects(state))

        broken_objects = [
            object_name
            for object_name in reception_objects
            if conditions.get(object_name)
            == BROKEN_CONDITION
        ]
        clean_objects = [
            object_name
            for object_name in reception_objects
            if object_name not in broken_objects
        ]

        total_by_type = Counter(
            product_type_from_name(object_name)
            for object_name in all_products
        )

        delivery_by_type = Counter()

        for assignment in assignments:
            delivery_by_type.update(assignment["products"])

        broken_by_type = Counter(
            product_type_from_name(object_name)
            for object_name in broken_objects
        )

        reception_assignment: List[Dict] = []

        if clean_objects:
            for product_type in PRODUCT_TYPES:
                storage_count = (
                    total_by_type[product_type]
                    - delivery_by_type[product_type]
                    - broken_by_type[product_type]
                )

                if storage_count > 0:
                    reception_assignment.append(
                        {
                            "targets": [f"{product_type}_grid"],
                            "products": {product_type: storage_count},
                        }
                    )

        if broken_objects:
            reception_assignment.append(
                {
                    "targets": [TRASH_LOCATION],
                    "products": dict(broken_by_type),
                }
            )

        fixed_targets = {
            object_name: TRASH_LOCATION
            for object_name in broken_objects
        }

        return (
            reception_assignment,
            fixed_targets,
        )

    def sample_parameters(self, state: TaskState) -> ProcessingParameters:
        available_products = (
            self._available_delivery_products(state)
        )

        maximum_delivery_count = min(
            self.max_deliveries,
            len(available_products),
            len(CUSTOMERS),
            MAX_CONCURRENT_DELIVERIES,
        )

        if maximum_delivery_count < 1:
            raise RuntimeError("No delivery can currently be created.")

        delivery_count = random.randint(1, maximum_delivery_count)

        assignments = self._build_deliveries(
            state=state,
            available_products=available_products,
            delivery_count=delivery_count,
        )

        current_reception_objects = self._reception_objects(state)

        all_products = _all_products(state)
        next_reception_count = random.randint(
            2,
            min(self.reception_size, len(all_products)),
        )

        next_reception_objects = random.sample(
            all_products, k=next_reception_count,
        )

        next_broken_count = random.randint(
            1,
            min(
                self.broken_count,
                next_reception_count,
            ),
        )

        next_broken_objects = random.sample(
            next_reception_objects,
            k=next_broken_count,
        )
        reception_offsets = random.sample(
            RECEPTION_OFFSETS,
            k=len(current_reception_objects),
        )
        return ProcessingParameters(
            tuple(DeliveryOrder.from_assignment(item) for item in assignments),
            tuple(current_reception_objects),
            tuple(next_reception_objects),
            tuple(next_broken_objects),
            tuple(reception_offsets),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: ProcessingParameters,
    ) -> List[BaseTaskStage]:
        assignments = parameters.assignment_dicts()
        reception_assignment, fixed_targets = self._build_reception_assignment(
            state,
            assignments,
        )
        objective_count = processing_objective_count(
            reception_assignment,
            fixed_targets,
            len(assignments),
        )
        stages = [
            AtLeastCompletedObjectivesStage(
                physical_assignment=reception_assignment,
                fixed_targets_by_object=fixed_targets,
                delivery_assignments=assignments,
                minimum_completed_goals=minimum,
                instruction=parameters.instruction if minimum == 1 else "none",
                reception_product_count=len(parameters.current_reception_objects),
                flag_answer=False,
            )
            for minimum in range(1, objective_count + 1)
        ]
        stages[0].entry_transition = ResetAdvancedDeliveryTransition(
            reception_objects=list(parameters.current_reception_objects),
            broken_objects=list(fixed_targets),
            reception_offsets=parameters.reception_offsets,
        )
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: ProcessingParameters,
    ) -> TaskState:
        state = save_assignments(state, parameters.assignment_dicts())
        next_reception = set(parameters.next_reception_objects)
        next_broken = set(parameters.next_broken_objects)

        locations = {}
        conditions = {}

        for object_name in _all_products(state):
            locations[object_name] = (
                RECEPTION_LOCATION
                if object_name in next_reception
                else _product_grid(object_name)
            )

            conditions[object_name] = (
                BROKEN_CONDITION
                if object_name in next_broken
                else CLEAN_CONDITION
            )

        state.relations[OBJECT_LOCATION_RELATION] = locations

        state.relations[OBJECT_CONDITION_RELATION] = conditions
        return state

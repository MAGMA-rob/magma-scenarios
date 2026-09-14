import math
import statistics
import torch
from typing import Dict, List, Any, Mapping, Sequence
from magma_core.simulation.goals import BaseGoal
from magma_core.simulation.data_structures import StageInput, UserInstruction, EmptyInstruction
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_core.simulation.data_structures import StageInput, UserInstruction, EmptyInstruction, Log
from .attributes import PRODUCT_TYPES, TRASHCAN_PLACEMENT_THRESHOLD
from magma_core.simulation.stage import (
    AskingBaseStage,
    BaseTaskStage,
    StageGlobalParameters,
)
from typing import Optional
from magma_core.simulation.stage import BaseStageEnvironmentTransition
from .utils import (
    get_pose,
    group_product_objects,
    product_type_from_name,
)

def _complete_product_counts(products: Dict[str, int]) -> Dict[str, int]:
    return {
        product_type: products.get(product_type, 0)
        for product_type in PRODUCT_TYPES
    }


def _validate_delivery_assignments(assignments: Sequence[Mapping[str, Any]]) -> List[Dict]:
    if not assignments:
        raise ValueError("assignments must contain at least one delivery.")

    normalized = []
    known_order_ids = set()
    for assignment in assignments:
        name = assignment.get("name")
        city = assignment.get("city")
        service = assignment.get("service")
        products = assignment.get("products")

        if not isinstance(name, str) or not name:
            raise ValueError("Each delivery requires a recipient name.")

        if not isinstance(city, str) or not city:
            raise ValueError("Each delivery requires a city.")

        if service not in {"priority", "standard"}:
            raise ValueError("Service must be 'priority' or 'standard'.")

        if not isinstance(products, dict) or not products:
            raise ValueError("Each delivery requires products.")


        normalized_products = {}

        for product_type, quantity in products.items():
            if product_type not in PRODUCT_TYPES:
                raise ValueError(f"Unknown product type: {product_type}.")

            if not isinstance(quantity, int) or quantity < 1:
                raise ValueError(f"Quantity for {product_type} must be >= 1.")

            normalized_products[product_type] = quantity


        order_id = assignment.get("order_id")

        if not isinstance(order_id, str) or not order_id:
            raise ValueError("Each delivery requires an order_id.")

        if order_id in known_order_ids:
            raise ValueError(f"Duplicate order_id: {order_id}.")

        known_order_ids.add(order_id)

        normalized.append(
            {
                "order_id": order_id,
                "name": name,
                "city": city,
                "service": service,
                "products": normalized_products,
            }
        )

    return normalized

def _shipping_label_content(log: Log):
    if log.function != "attach_shipping_label":
        return None

    content = log.content

    required_keys = {
        "order_id",
        "package",
        "bay",
        "name",
        "city",
        "products",
    }

    if not isinstance(content, dict):
        return None

    if set(content) != required_keys:
        return None

    if not isinstance(content["package"], str):
        return None

    if not isinstance(content["bay"], str):
        return None

    if not isinstance(content["name"], str):
        return None

    if not isinstance(content["city"], str):
        return None

    if not isinstance(content["products"], dict):
        return None

    if not isinstance(content["order_id"], str):
        return None

    return content

def _validated_delivery_count(assignments: Sequence[Mapping[str, Any]], logs: List[Log]) -> int:
    expected_by_order = {
        assignment["order_id"]: {
            "name": assignment["name"],
            "city": assignment["city"],
            "service": assignment["service"],
            "products": _complete_product_counts(
                assignment["products"]
            ),
        }
        for assignment in assignments
    }

    validated_orders = set()
    used_packages = set()

    for log in logs:
        content = _shipping_label_content(log)

        if content is None:
            continue

        order_id = content["order_id"]

        if order_id not in expected_by_order:
            continue

        expected = expected_by_order[order_id]

        package_name = content["package"]

        correct_package = package_name.startswith(f"{expected['service']}_")

        correct_recipient = (
            content["name"] == expected["name"]
            and content["city"] == expected["city"]
        )

        correct_products = (content["products"] == expected["products"])

        correct_bay = content["bay"].startswith(
            f"{expected['service']}_bay_"
        )

        if order_id in validated_orders:
            continue

        if (
            not correct_recipient
            or not correct_products
            or not correct_package
            or not correct_bay
        ):
            continue

        if (
            order_id in validated_orders
            or package_name in used_packages
        ):
            continue

        validated_orders.add(order_id)
        used_packages.add(package_name)


    return len(validated_orders)

class AtLeastAssignedProductCount(BaseGoal):
    """
    Hybrid placement goal:

    - regular products are interchangeable and counted by type;
    - fixed products must individually reach their assigned target.

    Assignment format:
    [
        {
            "targets": ["electronics_grid"],
            "products": {"electronics": 2},
        },
        {
            "targets": ["trashcan"],
            "products": {"drinks": 1},
        },
    ]

    Fixed targets format:
    {
        "drinks_3": "trashcan",
    }
    """

    def __init__(
        self,
        assignment: List[Dict],
        minimum: int,
        fixed_targets_by_object: Optional[Dict[str, str]] = None,
        thresh: float = 0.2,
    ) -> None:
        super().__init__(f"AtLeastAssignedProductCount",f"minimum={minimum}")

        if not assignment:
            raise ValueError("assignment must not be empty.")

        self.assignment = assignment
        self.minimum = minimum
        self.thresh = thresh
        self.fixed_targets_by_object = dict(fixed_targets_by_object or {})

        total_required = 0

        for request in self.assignment:
            targets = request.get("targets", [])
            products = request.get("products", {})

            if not targets:
                raise ValueError("Each assignment requires at least one target.")

            if not products:
                raise ValueError("Each assignment requires products.")

            for product_type, required_count in products.items():
                if product_type not in PRODUCT_TYPES:
                    raise ValueError(f"Unknown product type: {product_type}.")

                if (
                    not isinstance(required_count, int)
                    or required_count < 1
                ):
                    raise ValueError(f"Count for {product_type} must be >= 1.")

                total_required += required_count

        if minimum < 1 or minimum > total_required:
            raise ValueError(
                f"minimum must be between 1 and {total_required}, got {minimum}."
            )

        for object_name, target in self.fixed_targets_by_object.items():
            object_type = product_type_from_name(object_name)

            if object_type not in PRODUCT_TYPES:
                raise ValueError(f"Invalid fixed product instance: {object_name!r}.")

            target_is_represented = any(
                target in request["targets"]
                and object_type in request["products"]
                for request in self.assignment
            )

            if not target_is_represented:
                raise ValueError(
                    f"The fixed assignment {object_name!r} "
                    f"to {target!r} is not represented in "
                    "the general assignment."
                )

    def verify(self, obs: Dict) -> torch.Tensor:
        extra = obs["extra"]

        first_target = self.assignment[0]["targets"][0]

        if first_target not in extra:
            raise KeyError(f"Unknown assignment target: {first_target}.")

        reference_pose = get_pose(extra[first_target])

        nb_envs = 1 if reference_pose.ndim == 1 else reference_pose.shape[0]
        

        completed_products = torch.zeros(
            nb_envs,
            dtype=torch.int32,
            device=reference_pose.device,
        )

        objects_by_type = group_product_objects(extra)

        for request in self.assignment:
            targets = request["targets"]
            products = request["products"]

            best_target_count = torch.zeros_like(completed_products)

            for target in targets:
                if target not in extra:
                    raise KeyError(f"Unknown assignment target: {target}.")

                target_pose = get_pose(extra[target])
                placement_threshold = (
                    TRASHCAN_PLACEMENT_THRESHOLD
                    if target == "trashcan"
                    else self.thresh
                )

                target_count = torch.zeros_like(completed_products)

                for product_type, required_count in products.items():
                    fixed_objects = [
                        object_name for object_name, fixed_target
                        in self.fixed_targets_by_object.items()
                        if (
                            fixed_target == target
                            and product_type_from_name(object_name) == product_type
                        )
                    ]

                    fixed_slots = min(len(fixed_objects), required_count)

                    fixed_count = torch.zeros_like(completed_products)

                    for object_name in fixed_objects:
                        if object_name not in extra:
                            raise KeyError(f"Unknown fixed product: {object_name}.")

                        fixed_count += (
                            is_object_inside_target(
                                get_pose(extra[object_name]),
                                target_pose,
                                thresh=placement_threshold,
                                keep_tensor=True,
                            ).int().reshape(-1)
                        )

                    fixed_count = torch.minimum(
                        fixed_count,
                        torch.full_like(fixed_count, fixed_slots),
                    )

                    generic_required = (required_count - fixed_slots)

                    generic_count = torch.zeros_like(completed_products)

                    for object_name in objects_by_type[product_type]:
                        generic_count += (
                            is_object_inside_target(
                                get_pose(extra[object_name]),
                                target_pose,
                                thresh=placement_threshold,
                                keep_tensor=True,
                            ).int().reshape(-1)
                        )

                    generic_count = torch.minimum(
                        generic_count,
                        torch.full_like(generic_count, generic_required),
                    )

                    target_count += (fixed_count + generic_count)

                best_target_count = torch.maximum(
                    best_target_count,
                    target_count,
                )

            completed_products += best_target_count

        return (completed_products >= self.minimum).int()

class AtLeastCompletedObjectivesStage(BaseTaskStage):
    def __init__(
        self,
        minimum_completed_goals: int,
        instruction: str,
        flag_answer: bool,
        physical_assignment: Optional[List[Dict]] = None,
        fixed_targets_by_object: Optional[ Dict[str, str]] = None,
        delivery_assignments: Sequence[Mapping[str, Any]] = (),
        reception_product_count: int = 5,
        entry_transition: Optional[BaseStageEnvironmentTransition] = None,
        linked_to_prev: bool = False,
    ) -> None:
        self.physical_assignment = [
            dict(request)
            for request in (physical_assignment or [])
        ]
        self.fixed_targets_by_object = dict(fixed_targets_by_object or {})
        self.reception_product_count = reception_product_count
        self.delivery_assignments = (
            _validate_delivery_assignments(delivery_assignments)
            if delivery_assignments else []
        )

        physical_assignment = list(physical_assignment or [])
        fixed_targets_by_object = dict(fixed_targets_by_object or {})

        clean_assignment = [
            request
            for request in physical_assignment
            if "trashcan" not in request.get("targets", [])
        ]
        broken_assignment = [
            request
            for request in physical_assignment
            if "trashcan" in request.get("targets", [])
        ]

        goals: List[BaseGoal] = []
        objective_costs = []

        broken_count = len(fixed_targets_by_object)
        clean_count = max(0, reception_product_count - broken_count)

        if clean_assignment:
            clean_minimum = sum(
                sum(request["products"].values())
                for request in clean_assignment
            )
            goals.append(
                AtLeastAssignedProductCount(
                    assignment=clean_assignment,
                    minimum=clean_minimum,
                )
            )
            objective_costs.append(6 * clean_count)

        if broken_assignment:
            broken_minimum = sum(
                sum(request["products"].values())
                for request in broken_assignment
            )
            goals.append(
                AtLeastAssignedProductCount(
                    assignment=broken_assignment,
                    fixed_targets_by_object=fixed_targets_by_object,
                    minimum=broken_minimum,
                )
            )
            objective_costs.append(3 * broken_count)

        if goals:
            detection_cost, detection_remainder = divmod(6, len(goals))
            for index in range(len(goals)):
                objective_costs[index] += detection_cost + int(
                    index < detection_remainder
                )

        objective_costs.extend(
            2 * sum(delivery["products"].values())
            + len(delivery["products"])
            + 4
            for delivery in self.delivery_assignments
        )

        self.total_objectives = len(objective_costs)

        if self.total_objectives == 0:
            raise ValueError("At least one physical or delivery objective is required.")

        if (
            minimum_completed_goals < 1
            or minimum_completed_goals > self.total_objectives
        ):
            raise ValueError(
                f"minimum_completed_goals must be between 1 and {self.total_objectives}."
            )

        self.minimum_completed_goals = minimum_completed_goals

        self.target_tool_calls = math.ceil(
            statistics.median(objective_costs)
        )
        self.max_tool_calls = math.ceil(
            1.5 * max(objective_costs)
        )

        descriptions = []

        is_reception_transition = (
            entry_transition is not None
            and not self.delivery_assignments
        )

        if clean_assignment:
            descriptions.append(
                "return all viable reception products to storage"
            )

        if broken_assignment:
            descriptions.append(
                "discard all broken reception products"
            )

        descriptions.extend(
            f"complete delivery {assignment['order_id']}"
            for assignment in self.delivery_assignments
        )

        if is_reception_transition:
            stage_goal_description = (
                "A new delivery of returned products is unloaded "
                "in the reception area."
            )
        else:
            stage_goal_description = (
                f"Complete at least {minimum_completed_goals} of the "
                f"{self.total_objectives} objectives: "
                f"{'; '.join(descriptions)}."
            )

        super().__init__(
            goals=goals,
            stage_goal_description=stage_goal_description,
            stage_input=StageInput(
                instruction=(
                    UserInstruction(instruction)
                    if instruction != "none"
                    else EmptyInstruction()
                ),
                flag_answer_to_user=flag_answer,
                linked_to_prev=linked_to_prev,
            ),
            global_parameters=StageGlobalParameters(
                reset_at_end=False,
            ),
            entry_transition=entry_transition,
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "minimum_completed_goals": self.minimum_completed_goals,
            "instruction": (
                "none"
                if isinstance(self.stage_input.instruction, EmptyInstruction)
                else self.stage_input.instruction.get_content()
            ),
            "flag_answer": self.stage_input.flag_answer_to_user,
            "physical_assignment": self.physical_assignment,
            "fixed_targets_by_object": self.fixed_targets_by_object,
            "delivery_assignments": self.delivery_assignments,
            "reception_product_count": self.reception_product_count,
            "entry_transition": self.entry_transition,
            "linked_to_prev": self.stage_input.linked_to_prev,
        }

    def verif_log_completion(self, stage_log: List[Log], full_log: List[Log]) -> int:
        if not self.delivery_assignments:
            return 0

        return _validated_delivery_count(self.delivery_assignments, full_log)

    def _verif_env_completion(self, obs: Dict) -> torch.Tensor:
        goals = self.get_goals()

        if not goals:
            return super()._verif_env_completion(obs)

        results = [goal.verify(obs) for goal in goals]
        completed = torch.zeros_like(results[0])
        catastrophic_failure = torch.zeros_like(results[0], dtype=torch.bool)

        for result in results:
            completed += (result == 1).int()
            catastrophic_failure |= result == -1

        return torch.where(
            catastrophic_failure,
            torch.full_like(completed, -1),
            completed,
        )

    def combine_stage_completion(self, task_completion: int, log_completion: int) -> int:
        if (
            task_completion == -1
            or log_completion == -1
        ):
            return -1

        completed_objectives = log_completion + task_completion

        return int(completed_objectives >= self.minimum_completed_goals)


class AskDeliveryDetailsStage(AskingBaseStage):
    def __init__(
        self,
        assignment: Mapping[str, Any],
        linked_to_prev: bool = False,
    ) -> None:
        delivery = _validate_delivery_assignments([assignment])[0]
        self.assignment = delivery

        product_descriptions = [
            (
                f"{quantity} {product_type} product"
                if quantity == 1
                else f"{quantity} {product_type} products"
            )
            for product_type, quantity
            in delivery["products"].items()
        ]

        products = ", ".join(
            product_descriptions
        )

        question = (
            f"What are the recipient name, destination city, "
            f"and requested products for delivery "
            f"{delivery['order_id']}?"
        )

        answer = (
            f"Delivery {delivery['order_id']} is for "
            f"{delivery['name']} in {delivery['city']} "
            f"and requires {products}."
        )

        super().__init__(
            question=question,
            answer=answer,
            allow_tools_before_answer=False,
        )

        self.global_parameters.reset_at_end = False
        self.stage_input.linked_to_prev = linked_to_prev

        self.stage_goal_description = answer

    def _to_spec_arguments(self) -> Dict:
        return {
            "assignment": self.assignment,
            "linked_to_prev": self.stage_input.linked_to_prev,
        }

class AskDeliveryServiceCountsStage(AskingBaseStage):
    def __init__(
        self,
        assignments: Sequence[Mapping[str, Any]],
        linked_to_prev: bool = False,
    ) -> None:
        deliveries = (
            _validate_delivery_assignments(
                assignments
            )
        )
        self.assignments = deliveries

        delivery_ids = [
            assignment["order_id"]
            for assignment in deliveries
        ]

        question = "Which deliveries are you preparing right now?"

        answer = (
            "The deliveries currently being prepared are "
            f"{', '.join(delivery_ids)}."
            if len(delivery_ids) > 1
            else (
                "The delivery currently being prepared is "
                f"{delivery_ids[0]}."
            )
        )

        super().__init__(
            question=question,
            answer=answer,
            allow_tools_before_answer=False,
        )

        self.global_parameters.reset_at_end = False
        self.stage_input.linked_to_prev = linked_to_prev

        self.stage_goal_description = answer

    def _to_spec_arguments(self) -> Dict:
        return {
            "assignments": self.assignments,
            "linked_to_prev": self.stage_input.linked_to_prev,
        }


class AskStorageProductCountStage(AskingBaseStage):
    target_tool_calls = 3
    max_tool_calls = 3

    def __init__(self, product_type: str, storage_instances: List[str]) -> None:
        self.product_type = product_type
        self.storage_instances = storage_instances.copy()
        count = len(storage_instances)

        answer = (
            f"There are {count} {product_type} product "
            f"instance{'s' if count != 1 else ''} in storage."
        )

        super().__init__(
            question=(
                f"How many {product_type} product instances "
                "are currently in storage?"
            ),
            answer=answer,
            allow_tools_before_answer=True,
        )

        self.global_parameters.reset_at_end = False
        self.stage_goal_description = answer

    def _to_spec_arguments(self) -> Dict:
        return {
            "product_type": self.product_type,
            "storage_instances": self.storage_instances,
        }


class AskPendingReturnsCountStage(AskingBaseStage):
    target_tool_calls = 3
    max_tool_calls = 3

    def __init__(self, reception_objects: List[str]) -> None:
        self.reception_objects = reception_objects.copy()
        count = len(reception_objects)

        answer = (
            f"There {'is' if count == 1 else 'are'} {count} returned "
            f"product{'s' if count != 1 else ''} to process."
        )

        super().__init__(
            question=(
                "How many returned products are currently waiting "
                "to be processed?"
            ),
            answer=answer,
            allow_tools_before_answer=True,
        )

        self.global_parameters.reset_at_end = False
        self.stage_goal_description = answer

    def _to_spec_arguments(self) -> Dict:
        return {"reception_objects": self.reception_objects}

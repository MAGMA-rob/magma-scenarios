from typing import Dict, List

import torch

from magma_core.simulation.data_structures import (
    EmptyInstruction,
    Log,
    StageInput,
    UserInstruction,
)
from magma_core.simulation.goals import BaseGoal
from magma_core.simulation.stage import BaseTaskStage, StageErrorParameters, StageGlobalParameters
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_scenarios.templates.errors import OneShotToolFailureError

from .attributes import PACKAGE_NAMES, att


PACKAGE_HELD = 1


def _instruction(value: str):
    return EmptyInstruction() if value == "none" else UserInstruction(value)


def _pose(entry):
    return entry["pose"] if isinstance(entry, dict) else entry


def _validate_products(products: Dict[str, int]) -> Dict[str, int]:
    if not isinstance(products, dict) or not products:
        raise ValueError("products must be a non-empty dictionary")

    normalized = {}
    for product_type, count in products.items():
        if product_type not in att["product_type"]:
            raise ValueError(f"Unknown product type {product_type}")
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"Count for {product_type} must be >= 1")
        normalized[product_type] = count
    return normalized


def _validate_assignments(assignments: List[Dict]) -> List[Dict]:
    if not assignments:
        raise ValueError("assignments must contain at least one order")
    if len(assignments) > len(PACKAGE_NAMES):
        raise ValueError("There are more orders than available packages")

    normalized = []
    known_orders = set()
    for assignment in assignments:
        manufacturing_order = assignment.get("manufacturing_order")
        if not isinstance(manufacturing_order, str) or not manufacturing_order:
            raise ValueError("Each order needs a manufacturing_order")
        if manufacturing_order in known_orders:
            raise ValueError(f"Duplicate manufacturing order: {manufacturing_order}")
        known_orders.add(manufacturing_order)
        normalized.append(
            {
                "manufacturing_order": manufacturing_order,
                "products": _validate_products(assignment.get("products")),
            }
        )
    return normalized


def _complete_product_counts(products: Dict[str, int]) -> Dict[str, int]:
    return {
        product_type: products.get(product_type, 0)
        for product_type in att["product_type"]
    }


def _mark_content(log: Log):
    if log.function != "mark_package":
        return None
    content = log.content
    required_keys = {"package", "manufacturing_order", "products"}
    if not isinstance(content, dict) or set(content) != required_keys:
        return None
    if content["package"] not in PACKAGE_NAMES:
        return None
    if not isinstance(content["manufacturing_order"], str):
        return None
    if not isinstance(content["products"], dict):
        return None
    return content


def _validated_assignment_count(assignments: List[Dict], logs: List[Log]) -> int:
    expected_by_order = {
        assignment["manufacturing_order"]: _complete_product_counts(
            assignment["products"]
        )
        for assignment in assignments
    }
    validated_orders = set()
    used_packages = set()

    for log in logs:
        content = _mark_content(log)
        if content is None:
            continue

        manufacturing_order = content["manufacturing_order"]
        if manufacturing_order not in expected_by_order:
            continue

        package_name = content["package"]
        if (
            manufacturing_order in validated_orders
            or package_name in used_packages
            or content["products"] != expected_by_order[manufacturing_order]
        ):
            return -1

        validated_orders.add(manufacturing_order)
        used_packages.add(package_name)

    return len(validated_orders)


class _HeldPackageProductsGoal(BaseGoal):
    def __init__(self, products: Dict[str, int] | None, minimum: int) -> None:
        super().__init__(
            name="ProductsInHeldPackage",
            metadata=f"minimum={minimum}",
        )
        self.products = products
        self.minimum = minimum

    def verify(self, obs: Dict) -> torch.Tensor:
        extra = obs["extra"]
        reference_pose = _pose(extra[PACKAGE_NAMES[0]])
        nb_envs = 1 if reference_pose.ndim == 1 else reference_pose.shape[0]
        result = torch.zeros(
            nb_envs,
            dtype=torch.int32,
            device=reference_pose.device,
        )

        for package_name in PACKAGE_NAMES:
            package_entry = extra[package_name]
            package_pose = _pose(package_entry)
            held = package_entry["state"].reshape(-1).eq(PACKAGE_HELD)

            if self.products is None:
                product_count = torch.zeros_like(result)
                product_types = att["product_type"]
            else:
                product_count = torch.zeros_like(result)
                product_types = self.products

            for product_type in product_types:
                actual_count = torch.zeros_like(result)
                for object_name, entry in extra.items():
                    if not object_name.startswith(f"{product_type}_"):
                        continue
                    actual_count += is_object_inside_target(
                        _pose(entry),
                        package_pose,
                        keep_tensor=True,
                    ).int().reshape(-1)

                if self.products is None:
                    product_count += actual_count
                else:
                    product_count += torch.minimum(
                        actual_count,
                        torch.full_like(actual_count, self.products[product_type]),
                    )

            matches = product_count.eq(0) if self.products is None else product_count.ge(self.minimum)
            result = torch.maximum(result, (held & matches).int())

        return result


class AtLeastPackageCompletedStage(BaseTaskStage):
    def __init__(
        self,
        assignments: List[Dict],
        minimum: int,
        instruction: str,
        flag_answer_to_user: bool,
        move_to_packages_before_take: bool = False,
        wait_after_completion: bool = False,
        linked_to_prev: bool = False,
    ) -> None:
        self.assignments = _validate_assignments(assignments)
        self.minimum = minimum
        self.instruction = instruction
        self.flag_answer_to_user = flag_answer_to_user
        self.move_to_packages_before_take = move_to_packages_before_take
        self.wait_after_completion = wait_after_completion
        self.linked_to_prev = linked_to_prev
        if minimum < 1 or minimum > len(self.assignments):
            raise ValueError(
                f"minimum must be between 1 and {len(self.assignments)}, got {minimum}"
            )

        completed_order = self.assignments[minimum - 1]
        product_count = sum(completed_order["products"].values())
        self.target_tool_calls = (
            2 * product_count
            + 6
            + int(move_to_packages_before_take)
            + int(wait_after_completion)
            + int(flag_answer_to_user)
        )
        self.max_tool_calls = self.target_tool_calls + 4

        order_descriptions = []
        for assignment in self.assignments:
            recipe = ", ".join(
                f"{quantity} {product_type}"
                for product_type, quantity in assignment["products"].items()
            )
            order_descriptions.append(
                f"order {assignment['manufacturing_order']} with recipe {recipe}"
            )

        super().__init__(
            goals=[],
            stage_goal_description=(
                f"Complete at least {minimum} package(s) among: "
                f"{'; '.join(order_descriptions)}. Each completed order must "
                "use a distinct package and match its recipe exactly."
            ),
            stage_input=StageInput(
                instruction=_instruction(instruction),
                flag_answer_to_user=flag_answer_to_user,
                linked_to_prev=linked_to_prev,
            ),
            global_parameters=StageGlobalParameters(reset_at_end=False),
            error_parameters=StageErrorParameters(
                possible_errors=[
                    OneShotToolFailureError()
                ]
            ),
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "assignments": self.assignments,
            "minimum": self.minimum,
            "instruction": self.instruction,
            "flag_answer_to_user": self.flag_answer_to_user,
            "move_to_packages_before_take": self.move_to_packages_before_take,
            "wait_after_completion": self.wait_after_completion,
            "linked_to_prev": self.linked_to_prev,
        }

    def verif_log_completion(self, stage_log: List[Log], full_log: List[Log]) -> int:
        count = _validated_assignment_count(self.assignments, full_log)
        if count < 0:
            return -1
        return int(count >= self.minimum)


class EmptyPackageTakenStage(BaseTaskStage):
    def __init__(
        self,
        instruction: str = "none",
        flag_answer_to_user: bool = False,
        replace_held_package: bool = False,
        linked_to_prev: bool = False,
    ) -> None:
        self.instruction = instruction
        self.flag_answer_to_user = flag_answer_to_user
        self.replace_held_package = replace_held_package
        self.linked_to_prev = linked_to_prev
        self.target_tool_calls = 4 if replace_held_package else 2
        self.max_tool_calls = self.target_tool_calls + 4
        super().__init__(
            goals=[_HeldPackageProductsGoal(products=None, minimum=0)],
            stage_goal_description="Hold any empty package.",
            stage_input=StageInput(
                instruction=_instruction(instruction),
                flag_answer_to_user=flag_answer_to_user,
                linked_to_prev=linked_to_prev,
            ),
            global_parameters=StageGlobalParameters(reset_at_end=False),
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "instruction": self.instruction,
            "flag_answer_to_user": self.flag_answer_to_user,
            "replace_held_package": self.replace_held_package,
            "linked_to_prev": self.linked_to_prev,
        }


class AtLeastProductsInHeldPackageStage(BaseTaskStage):
    def __init__(
        self,
        products: Dict[str, int],
        minimum: int,
        instruction: str = "none",
        flag_answer_to_user: bool = False,
        inspect_empty_package: bool = False,
        resume_interrupted_package: bool = False,
    ) -> None:
        if inspect_empty_package and resume_interrupted_package:
            raise ValueError(
                "A product stage cannot both inspect a new package and resume "
                "an interrupted package"
            )
        self.target_tool_calls = 2
        if inspect_empty_package:
            self.target_tool_calls += 2
        if resume_interrupted_package:
            self.target_tool_calls += 3
        self.max_tool_calls = self.target_tool_calls + 4
        self.products = _validate_products(products)
        self.instruction = instruction
        self.flag_answer_to_user = flag_answer_to_user
        self.inspect_empty_package = inspect_empty_package
        self.resume_interrupted_package = resume_interrupted_package
        total = sum(self.products.values())
        if minimum < 1 or minimum > total:
            raise ValueError(f"minimum must be between 1 and {total}, got {minimum}")
        self.minimum = minimum
        recipe = ", ".join(
            f"{quantity} {product_type}"
            for product_type, quantity in self.products.items()
        )

        super().__init__(
            goals=[_HeldPackageProductsGoal(self.products, minimum)],
            stage_goal_description=(
                f"Hold a package containing at least {minimum} requested "
                f"product instance(s) from the recipe: {recipe}. Counts for "
                "each product type are capped by this recipe."
            ),
            stage_input=StageInput(
                instruction=_instruction(instruction),
                flag_answer_to_user=flag_answer_to_user,
            ),
            global_parameters=StageGlobalParameters(reset_at_end=False),
            error_parameters=StageErrorParameters(
                possible_errors=[
                    OneShotToolFailureError()
                ]
            ),
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "products": self.products.copy(),
            "minimum": self.minimum,
            "instruction": self.instruction,
            "flag_answer_to_user": self.flag_answer_to_user,
            "inspect_empty_package": self.inspect_empty_package,
            "resume_interrupted_package": self.resume_interrupted_package,
        }


class ValidatePackageStage(BaseTaskStage):
    def __init__(
        self,
        assignment: Dict,
        instruction: str = "none",
        flag_answer_to_user: bool = False,
    ) -> None:
        self.instruction = instruction
        self.flag_answer_to_user = flag_answer_to_user
        self.target_tool_calls = 3 + int(flag_answer_to_user)
        self.max_tool_calls = self.target_tool_calls + 4
        self.assignment = _validate_assignments([assignment])[0]
        super().__init__(
            goals=[],
            stage_goal_description=(
                f"Drop and mark a package for order "
                f"{self.assignment['manufacturing_order']} with its exact recipe."
            ),
            stage_input=StageInput(
                instruction=_instruction(instruction),
                flag_answer_to_user=flag_answer_to_user,
            ),
            global_parameters=StageGlobalParameters(reset_at_end=False),
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "assignment": {
                "manufacturing_order": self.assignment["manufacturing_order"],
                "products": self.assignment["products"].copy(),
            },
            "instruction": self.instruction,
            "flag_answer_to_user": self.flag_answer_to_user,
        }

    def verif_log_completion(self, stage_log: List[Log], full_log: List[Log]) -> int:
        mark_logs = [log for log in stage_log if log.function == "mark_package"]
        if not mark_logs:
            return 0
        if len(mark_logs) != 1:
            return -1

        content = _mark_content(mark_logs[0])
        expected_products = _complete_product_counts(self.assignment["products"])
        if content is None:
            return -1
        if content["manufacturing_order"] != self.assignment["manufacturing_order"]:
            return -1
        if content["products"] != expected_products:
            return -1
        return 1

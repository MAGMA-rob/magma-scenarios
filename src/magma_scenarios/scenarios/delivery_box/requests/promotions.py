import random
from collections import Counter

from magma_core.simulation.constraints import BaseConstraint
from magma_core.simulation.state import TaskState
from magma_scenarios.templates.requests.interact_request import (
    BaseConstraintRequest,
    ConstraintParameters,
)

from .constants import (
    ACTIVE_PROMOTION_KEY,
    GIFT_PROMOTION,
    REQUESTS_SINCE_PROMOTION_KEY,
    THIRD_PRODUCT_PROMOTION,
)


def active_promotion(state: TaskState) -> dict | None:
    promotion = state.properties.get(ACTIVE_PROMOTION_KEY)
    return promotion if isinstance(promotion, dict) else None


def increment_promotion_age(state: TaskState) -> None:
    if active_promotion(state) is None:
        state.properties[REQUESTS_SINCE_PROMOTION_KEY] = 0
        return
    state.properties[REQUESTS_SINCE_PROMOTION_KEY] = (
        state.properties.get(REQUESTS_SINCE_PROMOTION_KEY, 0) + 1
    )


class GiftPromotionConstraint(BaseConstraint):
    def __init__(self, product: str) -> None:
        super().__init__()
        self.product = product

    def apply(self, state: TaskState):
        super().apply(state)
        state.properties[ACTIVE_PROMOTION_KEY] = {
            "kind": GIFT_PROMOTION,
            "product": self.product,
        }
        state.properties[REQUESTS_SINCE_PROMOTION_KEY] = 0

    def outdated(self, state: TaskState) -> bool:
        return active_promotion(state) != {
            "kind": GIFT_PROMOTION,
            "product": self.product,
        }


class ThirdProductPromotionConstraint(BaseConstraint):
    def apply(self, state: TaskState):
        super().apply(state)
        state.properties[ACTIVE_PROMOTION_KEY] = {
            "kind": THIRD_PRODUCT_PROMOTION,
        }
        state.properties[REQUESTS_SINCE_PROMOTION_KEY] = 0

    def outdated(self, state: TaskState) -> bool:
        promotion = active_promotion(state)
        return promotion is None or promotion.get("kind") != THIRD_PRODUCT_PROMOTION


class ForgetPromotionConstraint(BaseConstraint):
    def apply(self, state: TaskState):
        super().apply(state)
        state.properties.pop(ACTIVE_PROMOTION_KEY, None)
        state.properties[REQUESTS_SINCE_PROMOTION_KEY] = 0

    def outdated(self, state: TaskState) -> bool:
        return active_promotion(state) is not None


class PromotionRequest(BaseConstraintRequest):
    """Common semantic trace for promotion-rule changes."""


class AlwaysAddGiftRequest(PromotionRequest):
    def sampling_weight(self, state: TaskState) -> float:
        if active_promotion(state) is not None:
            return 0

        available_counts = Counter(
            object_name.rsplit("_", 1)[0]
            for object_name in state.properties.get("table_objects", [])
        )
        return float(any(quantity >= 2 for quantity in available_counts.values()))

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        available_counts = Counter(
            object_name.rsplit("_", 1)[0]
            for object_name in state.properties.get("table_objects", [])
        )
        candidates = [
            product_type
            for product_type, quantity in available_counts.items()
            if quantity >= 2
        ]
        if not candidates:
            raise RuntimeError(
                "No product type has at least two instances available for the "
                "gift promotion."
            )

        gift_product = random.choice(candidates)
        return ConstraintParameters(
            [GiftPromotionConstraint(gift_product)],
            f"For each future order, add one {gift_product} as a free gift.",
        )


class AddThirdProductRequest(PromotionRequest):
    def sampling_weight(self, state: TaskState) -> float:
        if active_promotion(state) is not None:
            return 0
        return float(bool(state.attributes.get("product_type", [])))

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        return ConstraintParameters(
            [ThirdProductPromotionConstraint()],
            (
                "For every future order that initially contains exactly two units "
                "of the same product, add a third unit of that product for free."
            ),
        )


class ForgetRuleRequest(PromotionRequest):
    def sampling_weight(self, state: TaskState) -> float:
        if active_promotion(state) is None:
            return 0
        requests_since_promotion = state.properties.get(
            REQUESTS_SINCE_PROMOTION_KEY,
            0,
        )
        return 0.3 * max(0, requests_since_promotion)

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        promotion = active_promotion(state)
        if promotion is None:
            raise RuntimeError("There is no active promotion to forget.")
        return ConstraintParameters(
            [ForgetPromotionConstraint()],
            "Forget the currently active promotion rule.",
        )

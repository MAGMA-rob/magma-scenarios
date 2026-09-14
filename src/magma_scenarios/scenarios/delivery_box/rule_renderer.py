from typing import List

from magma_core.simulation.state import RuleRenderer, TaskState

from .requests.constants import (
    ACTIVE_PROMOTION_KEY,
    GIFT_PROMOTION,
    THIRD_PRODUCT_PROMOTION,
)


class DeliveryRuleRenderer(RuleRenderer):
    """Project the currently active promotion into one canonical rule."""

    def rules(self, state: TaskState) -> List[str]:
        promotion = state.properties.get(ACTIVE_PROMOTION_KEY)
        if not isinstance(promotion, dict):
            return []
        if promotion.get("kind") == GIFT_PROMOTION:
            return [
                f"Add one {promotion['product']} as a free gift to every order."
            ]
        if promotion.get("kind") == THIRD_PRODUCT_PROMOTION:
            return [
                "When an order contains two units of one product, add a third "
                "unit for free."
            ]
        raise ValueError(f"Unsupported delivery promotion: {promotion!r}.")


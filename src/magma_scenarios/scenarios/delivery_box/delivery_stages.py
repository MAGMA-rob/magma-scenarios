from typing import List

from .attributes import MAX_NB_PER_RECIPE, att

from magma_core.base.stage import BaseTaskStage
from magma_core.base.goals import AtLeastCountAt, BaseGoal, ExactCountAt, MaxAt
from magma_core.base.data_structures import UserInstruction, Log, Situation


def _product_instances(product_name: str) -> List[str]:
    return [f"{product_name}_1", f"{product_name}_2"]


def _build_goals_for_product(product_name: str, count: int) -> List[BaseGoal]:
    if count < 0 or count > MAX_NB_PER_RECIPE:
        raise ValueError(
            f"Recipe for '{product_name}' must be between 0 and {MAX_NB_PER_RECIPE}, got {count}."
        )

    objects = _product_instances(product_name)

    if count == 0:
        return [MaxAt(objects, "container", 0)]

    if count == 1:
        return [ExactCountAt(objects, "container", 1)]

    return [AtLeastCountAt(objects, "container", 2)]


def _build_recipe_goals(recipe: List[str]) -> List[BaseGoal]:
    recipe_counts = {product_name: 0 for product_name in att["product_type"]}

    for product_name in recipe:
        if product_name not in recipe_counts:
            raise ValueError(
                f"Unknown product '{product_name}'. Expected one of {att['product_type']}."
            )
        recipe_counts[product_name] += 1

    goals: List[BaseGoal] = []
    for product_name, count in recipe_counts.items():
        goals.extend(_build_goals_for_product(product_name, count))

    return goals


class CycleStage(BaseTaskStage):
    """
   Task STage to launch a cycle according to a reference, a manufacturing order and a number of delivery.
    """

    target_steps = 1
    acceptance_steps = 1

    def __init__(self, recipe : List[str], manufacturing_order : str, deliveries : int, instruction : str, flag_answer_to_user : bool) -> None:
        self.recipe = recipe
        self.manufacturing_order = manufacturing_order
        self.deliveries = deliveries

        self.situation = Situation(
            memory = ["You must make a delivery box by packing inside some objects."],
            preserved_memory_indices=[0],
            attributes=att,
            instruction=UserInstruction(instruction),
            flag_answer_to_user=flag_answer_to_user
        )

        super().__init__(
            _build_recipe_goals(recipe),
            True,
            f"The goal of the stage is to launch a cycle respecting recipe={self.recipe}, delivery_number={self.deliveries}, manufacturing_order={self.manufacturing_order}",
        )

    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if len(stage_log) == 0:
            return 0
        if stage_log[-1].content == (self.deliveries, self.manufacturing_order):
            return 1
        return -1

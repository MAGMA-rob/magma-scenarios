from typing import Any, List, Dict, Tuple

from .attributes import att

from magma_core.base.stage import BaseTaskStage
from magma_core.base.goals import At
from magma_core.base.data_structures import UserInstruction, Log, Situation

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

        super().__init__([At(o,"container") for o in recipe], True, f"The goal of the stage is to launch a cycle respecting recipe={self.recipe}, delivery_number={self.deliveries}, manufacturing_order={self.manufacturing_order}")

    
    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if len(stage_log) == 0:
            return 0
        if stage_log[-1].content == (self.deliveries, self.manufacturing_order):
            return 1
        return -1
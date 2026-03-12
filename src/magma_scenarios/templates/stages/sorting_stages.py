
from magma_core.base.stage import BaseTaskStage
from magma_core.base.data_structures import Situation, Instruction, Log

from typing import List, Dict, Literal, Optional

class MissingInformationStage(BaseTaskStage):
    """
    This stage allows to handle phase where users asks for an action but the agent does not have all needed informations.

    """

    target_steps = 1
    acceptance_steps = 0

    def __init__(
            self,
            instruction : Instruction,
            answer : str,
            memory : List[str],
            attributes : Dict
        ) -> None:
        super().__init__([], True, f"The stage of the goal is to ensure that the model correctly asks : {answer}")
        
        self.situation = Situation(
            memory=memory,
            preserved_memory_indices=[],
            instruction=instruction,
            attributes=attributes,
            flag_answer_to_user=False
        )
        self.verification_prompt = answer

class ForbiddenElemStage(BaseTaskStage):
    """
    This stage allows to handle phase where users asks for an action but it is forbidden.
    """

    target_steps = 1
    acceptance_steps = 0

    def __init__(
            self,
            instruction : Instruction,
            answer : str,
            memory : List[str],
            attributes : Dict
        ) -> None:
        super().__init__([], True, f"The stage of the goal is to ensure that : {answer}")
        
        self.situation = Situation(
            memory=memory,
            preserved_memory_indices=[],
            instruction=instruction,
            attributes=attributes,
            flag_answer_to_user=False
        )
        self.verification_prompt = answer


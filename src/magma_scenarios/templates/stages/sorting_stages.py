
from magma_core.base.stage import BaseTaskStage
from magma_core.base.data_structures import Situation, Instruction, Log, EmptyInstruction
from magma_core.base.goals import NotAt, At

from typing import List, Dict, Literal, Optional

class MissingInformationStage(BaseTaskStage):
    """
    Text-only stage used when the agent must ask for missing sorting details.
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
    Text-only stage used when the request involves forbidden objects or areas.
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


class Cycle(BaseTaskStage):
    """Execution stage for a sorting cycle with direct placement and forbidden-area checks."""

    target_steps = 1
    acceptance_steps = 1

    def __init__(
            self,
            assignment : Dict[str,str],
            known_areas : List[str],
            flag_answer : bool,
            manu_order : Optional[str] = None,
            instruction : Instruction = EmptyInstruction(),
        ) -> None:
        """
        Docstring for __init__
        
        :param instruction: The instruction to give to the user. If not specified, it will create an empty instruction (supposing that you gave previosuly an instruction to launch multiple cycle)
        :param instruction: str.
        :param assignement: Dict of assignment. Objects as key and area as value. Objects which should not be manipulated must have 'none' as value.
        :type assignement: Dict[str, str]
        :param all_area: The list of all existing area.
        :type all_area: List[str]
        :param manu_order: The optional desired manufacturing order.
        :type manu_order: Optional[str]
        """

        goals = []
        for obj_name, associated_area in assignment.items():     
            forbidden_area = []
            for area in known_areas:
                if area == associated_area:
                    goals.append(At(obj_name, area))
                else:
                    forbidden_area.append(area)
            if forbidden_area:
                goals.append(NotAt(obj_name, forbidden_area, True))

        super().__init__(goals, 
            reset_at_end=True,
            stage_goal_description=f"The goal of this stage is to sort all objects according to the assignment provided by the user and the memory : {assignment}")
        self.manu_order = manu_order

        self.situation = Situation(
            memory=["You are in charge of sorting objects in a factory."],
            preserved_memory_indices=[0],
            instruction=instruction,
            attributes={
                "objects" : list(assignment.keys()),
                "target_areas" : known_areas,
            },
            flag_answer_to_user=flag_answer
        )

    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if self.manu_order is None: return 1
        if len(stage_log) == 0: return 0
        if stage_log[-1].content == self.manu_order: return 1
        return -1

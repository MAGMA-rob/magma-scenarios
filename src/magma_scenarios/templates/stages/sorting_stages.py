
from magma_core.simulation.stage import BaseTaskStage, StageGlobalParameters
from typing import Dict

from magma_core.simulation.data_structures import Instruction, StageInput


class MissingInformationStage(BaseTaskStage):
    """
    [TEXT-ONLY]
    Text-only stage used when the agent must ask for missing sorting details.
    """

    target_tool_calls = 1
    max_tool_calls = 1


    def __init__(
            self,
            instruction : Instruction,
            answer : str,
        ) -> None:
        super().__init__(
            [],
            f"The stage of the goal is to ensure that the model correctly asks : {answer}",
            StageInput(
                instruction,
                False,
            ),
            StageGlobalParameters(
                reset_at_end=True,
                verification_prompt=answer
            )
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "instruction": self.get_stage_input().instruction,
            "answer": self.get_verification_prompt(),
        }

class ForbiddenElemStage(BaseTaskStage):
    """
    [TEXT-ONLY]
    Text-only stage used when the request involves forbidden objects or areas.
    """


    target_tool_calls = 1
    max_tool_calls = 1

    def __init__(
            self,
            instruction : Instruction,
            answer : str,
        ) -> None:
        super().__init__(
            [],
            f"The stage of the goal is to ensure that : {answer}",
            StageInput(instruction,False),
            StageGlobalParameters(reset_at_end=True,verification_prompt=answer)
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "instruction": self.get_stage_input().instruction,
            "answer": self.get_verification_prompt(),
        }

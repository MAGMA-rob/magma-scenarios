from typing import List, Dict

from magma_core.simulation.stage.stage_template import AskingBaseStage
from magma_core.simulation.stage import (
    BaseTaskStage,
    StageErrorParameters,
    StageGlobalParameters,
)
from magma_core.simulation.data_structures import (
    EmptyInstruction, UserInstruction,
    Instruction, Log, StageInput
)
from magma_core.simulation.goals import AtLeastCountAt
from magma_core.utils.text_utils import join_with_and

from .laundry_errors import GraspClothesFailureError

class LoadClotheStage(BaseTaskStage):
    """
    Stage to load desired clothes in the machine
    """

    target_tool_calls = 2
    max_tool_calls = 3

    def __init__(
        self,
        n: int,
        clothes_to_load: List[str],
        instruction: Instruction = EmptyInstruction(),
        linked_to_prev: bool | None = None,
    ) -> None:
        self.n = n
        self.clothes_to_load = clothes_to_load.copy()
        super().__init__(
            [AtLeastCountAt(clothes_to_load,"washing_machine_basket",n)],
            f"The goal of this stage is to have {n} clothes from the list {clothes_to_load} in the washing machine",
            StageInput(
                instruction = instruction,
                flag_answer_to_user=False,
                linked_to_prev=linked_to_prev,
            ),
            error_parameters=StageErrorParameters(
                possible_errors=[GraspClothesFailureError(clothes_to_load)]
            ),
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "n": self.n,
            "clothes_to_load": self.clothes_to_load.copy(),
            "instruction": self.get_stage_input().instruction,
            "linked_to_prev": self.get_stage_input().linked_to_prev,
        }

class WashStage(BaseTaskStage):
    """
    Stage for putting detergent and then launching the washing machine.
    """

    target_tool_calls = 4
    max_tool_calls = 4

    def __init__(
            self,
            target_detergent: str,
            target_clothes: list,
        flag_answer: bool = True,
        ) -> None:
        self.target_tool_calls = 4 if flag_answer else 3
        super().__init__([],
        f"The goal of this stage is to put the target detergent {target_detergent} into the washing machine then launch the cycle",
        StageInput(
            instruction= EmptyInstruction(),
            flag_answer_to_user=flag_answer,
            linked_to_prev=True ## It will be set to true automatically
        ),
        StageGlobalParameters(
            reset_at_end=True
        ))

        self.to_clean = target_clothes
        self.target_detergent = target_detergent
        self.flag_answer = flag_answer

    def _to_spec_arguments(self) -> Dict:
        return {
            "target_detergent": self.target_detergent,
            "target_clothes": self.to_clean.copy(),
            "flag_answer": self.flag_answer,
        }

    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if len(stage_log) == 0:
            return 0
        
        if len(stage_log[-1].content["clothes" ]) != len(self.to_clean):
            return -1

        for clothe in self.to_clean:
            if not clothe in stage_log[-1].content["clothes"]:
                return -1

        if len(stage_log[-1].content["detergent"]) > 1 :
            return -1
        
        if self.target_detergent not in stage_log[-1].content["detergent"] :
            return -1
        
        return 1

class RefuseLaundryStage(BaseTaskStage):
    target_tool_calls = 1
    max_tool_calls = 1

    def __init__(self, instruction, verif_prompt : str) -> None:
        super().__init__(
            [],
            "The goal of this stage is to refuse the user request.",
            StageInput(
                flag_answer_to_user=False,
                instruction=UserInstruction(instruction),
            ),
            StageGlobalParameters(
                verification_prompt=verif_prompt
            )
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "instruction": self.get_stage_input().instruction.get_content(),
            "verif_prompt": self.get_verification_prompt(),
        }

class AskClothesDetergentStage(AskingBaseStage):
    """
    Q&A stage testing clothes-to-detergent.
    The agent must list clothes belonging to a given detergent from state relations.
    No tools are allowed; answer is fully derived from clothes_to_detergent mapping.
    """
    def __init__(
        self,
        clothes_to_detergent : Dict[str, str],
        target_detergent : str ) -> None :

        question = f"Which clothes use '{target_detergent}'?"
        clothes = [ cloth for cloth,detergent in clothes_to_detergent.items() 
            if detergent == target_detergent ]
            
        if len(clothes) == 0:
            answer = f"No clothes use {target_detergent}."
        else :
            answer =  f"Clothes using '{target_detergent}' are: {', '.join(clothes)}"

        super().__init__(
            question=question,
            answer=answer,
            allow_tools_before_answer=False
        )
        self.clothes_to_detergent = clothes_to_detergent.copy()
        self.target_detergent = target_detergent

    def _to_spec_arguments(self) -> Dict:
        return {
            "clothes_to_detergent": self.clothes_to_detergent.copy(),
            "target_detergent": self.target_detergent,
        }


def _plural(word: str, values: List[str]) -> str:
    return word + "s" if len(values) > 1 else word


class AskClothesDetergentStageInverse(AskingBaseStage):

    def __init__(
        self,
        clothes_to_detergent: Dict[str, str],
        clothes: List[str]
    ) -> None:

        question = (
            f"What {_plural('detergent', clothes)} can wash {join_with_and(clothes)}?"
        )

        grouped_clothes = {}

        for cloth in clothes:
            detergent = clothes_to_detergent.get(cloth)

            if detergent is None:
                continue

            grouped_clothes.setdefault(detergent, []).append(cloth)

        if len(grouped_clothes) == 0:
            answer = (
                f"No detergent is associated with {join_with_and(clothes)}."
            )
        else:
            answer = ", ".join(
                f"{join_with_and(clothes_list)} can be washed with {detergent}"
                for detergent, clothes_list in grouped_clothes.items()
            )

        super().__init__(
            question=question,
            answer=answer,
            allow_tools_before_answer=False
        )
        self.clothes_to_detergent = clothes_to_detergent.copy()
        self.clothes = clothes.copy()

    def _to_spec_arguments(self) -> Dict:
        return {
            "clothes_to_detergent": self.clothes_to_detergent.copy(),
            "clothes": self.clothes.copy(),
        }

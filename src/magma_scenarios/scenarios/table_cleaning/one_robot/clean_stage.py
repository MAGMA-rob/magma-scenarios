from typing import Dict, List, Optional

from magma_core.simulation.data_structures import (
    UserInstruction, EmptyInstruction,
    Log, StageInput
)
from magma_core.simulation.stage import (
    AskingBaseStage,
    BaseTaskStage,
    StageErrorParameters,
    StageGlobalParameters,
    BaseStageEnvironmentTransition,
)

from ..common import (
    AtLeastTableRequirementCount,
    AtLeastTypedAssignedCount,
    AtMostTableObjectCount,
)
from ..common.attributes import (
    CLEAN_STATE,
    DIRTY_STATE,
    dishware,
    food,
)
from ..common.errors import (
    GraspClearTableFailureError,
    GraspItemsFailureError,
    GraspSetTableFailureError,
    MaskClearTableError,
    MaskItemsError,
    MaskSetTableError,
)


def _verify_put_destinations(stage_log: List[Log]) -> int:
    for log in stage_log:
        if log.function != "put":
            continue

        content = log.content or {}
        obj = content.get("object")
        target = content.get("target")
        state = content.get("state")

        if obj in food:
            if state == DIRTY_STATE and target != "trashcan":
                return -1
            if state == CLEAN_STATE and target not in ("food_storage", "table"):
                return -1

        elif obj in dishware:
            if state == DIRTY_STATE and target != "washing_machine":
                return -1
            if state == CLEAN_STATE and target not in ("dish_storage", "table"):
                return -1

    return 0


class PlaceObjectsStage(BaseTaskStage):
    """
    Goal: place at least n objects according to type counts and fixed targets.
    """
    target_tool_calls = 3
    max_tool_calls = 4

    def __init__(
            self,
            n: int,
            required_type_counts: Dict[str, Dict[str, int]],
            fixed_targets_by_object: Dict[str, str],
            instruction: str,
            flag_answer: bool,
            thresh: float = 0.2,
        ) -> None:
        self.required_type_counts = required_type_counts
        self.fixed_targets_by_object = fixed_targets_by_object
        self.n = n
        self.thresh = thresh

        total_targets = sum(
            count
            for type_counts in required_type_counts.values()
            for count in type_counts.values()
        )

        if n < 1 or n > total_targets:
            raise ValueError(
                f"n must be between 1 and {total_targets}, got {n}"
            )

        goal = AtLeastTypedAssignedCount(
            required_type_counts=required_type_counts,
            fixed_targets_by_object=fixed_targets_by_object,
            minimum=n,
            thresh=thresh,
        )

        assignment_parts = []

        for location, type_counts in required_type_counts.items():
            location_parts = [
                f"{count} {obj_type}"
                for obj_type, count in type_counts.items()
                if count > 0
            ]

            if location_parts:
                assignment_parts.append(
                    f"{', '.join(location_parts)} in {location}"
                )

        fixed_parts = [
            f"{obj} in {target}"
            for obj, target in fixed_targets_by_object.items()
        ]

        description_parts = []

        if assignment_parts:
            description_parts.append(
                "required type counts: " + "; ".join(assignment_parts)
            )

        if fixed_parts:
            description_parts.append(
                "fixed targets: " + "; ".join(fixed_parts)
            )

        stage_goal_description = (
            "The goal is to place objects according to "
            + " and ".join(description_parts)
            + "."
        )

        possible_errors = [
            MaskItemsError(
                max_masking=3,
                required_type_counts=required_type_counts,
                fixed_targets_by_object=fixed_targets_by_object,
                minimum=n,
                thresh=thresh,
            ),
            GraspItemsFailureError(
                max_impossible=3,
                required_type_counts=required_type_counts,
                fixed_targets_by_object=fixed_targets_by_object,
                minimum=n,
                thresh=thresh,
            ),
        ]

        super().__init__(
            [goal],
            stage_goal_description,
            StageInput(
                instruction=UserInstruction(instruction) if instruction != "none" else EmptyInstruction(),
                flag_answer_to_user=flag_answer,
            ),
            error_parameters=StageErrorParameters(
                possible_errors=possible_errors,
            ),
            )

    def _to_spec_arguments(self) -> Dict:
        return {
            "n": self.n,
            "required_type_counts": self.required_type_counts,
            "fixed_targets_by_object": self.fixed_targets_by_object,
            "instruction": (
                "none" if isinstance(self.stage_input.instruction, EmptyInstruction)
                else self.stage_input.instruction.get_content()
            ),
            "flag_answer": self.stage_input.flag_answer_to_user,
            "thresh": self.thresh,
        }

    def _object_type(self, obj: str) -> str:
        return obj.rsplit("_", 1)[0]


    def verif_log_completion(self, stage_log: List[Log], full_log: List[Log]) -> int:
        return _verify_put_destinations(stage_log)


    def combine_stage_completion(self, task_completion: int, log_completion: int) -> int:
        if log_completion == -1:
            return -1

        if task_completion == 1:
            return 1

        return 0

class ClearTableStage(BaseTaskStage):
    target_tool_calls = 4
    max_tool_calls = 5

    def __init__(
        self,
        active_objects: List[str],
        maximum_remaining: int,
        instruction: str,
        flag_answer: bool,
        entry_transition: Optional[BaseStageEnvironmentTransition] = None,
        thresh: float = 0.2,
        target_tool_calls: int = 4,
        max_tool_calls: int = 5,
    ) -> None:
        self.target_tool_calls = target_tool_calls
        self.max_tool_calls = max_tool_calls
        self.active_objects = list(active_objects)
        self.maximum_remaining = maximum_remaining
        self.thresh = thresh
        possible_errors = [
            MaskClearTableError(
                active_objects=active_objects,
                maximum_remaining=maximum_remaining,
                max_masking=3,
                thresh=thresh,
            ),
            GraspClearTableFailureError(
                active_objects=active_objects,
                maximum_remaining=maximum_remaining,
                max_impossible=3,
                thresh=thresh,
            ),
        ]

        super().__init__(
            goals=[
                AtMostTableObjectCount(
                    active_objects=active_objects,
                    maximum=maximum_remaining,
                    thresh=thresh,
                )
            ],
            stage_goal_description=(
                "The goal is to leave at most "
                f"{maximum_remaining} serving objects on the table."
            ),
            stage_input=StageInput(
                instruction=(
                    UserInstruction(instruction)
                    if instruction != "none"
                    else EmptyInstruction()
                ),
                flag_answer_to_user=flag_answer,
            ),
            error_parameters=StageErrorParameters(possible_errors=possible_errors),
            entry_transition=entry_transition,
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "active_objects": self.active_objects,
            "maximum_remaining": self.maximum_remaining,
            "instruction": (
                "none" if isinstance(self.stage_input.instruction, EmptyInstruction)
                else self.stage_input.instruction.get_content()
            ),
            "flag_answer": self.stage_input.flag_answer_to_user,
            "entry_transition": self.entry_transition,
            "thresh": self.thresh,
            "target_tool_calls": self.target_tool_calls,
            "max_tool_calls": self.max_tool_calls,
        }

    def verif_log_completion(self, stage_log: List[Log], full_log: List[Log]) -> int:
        return _verify_put_destinations(stage_log)

    def combine_stage_completion(self, task_completion: int, log_completion: int) -> int:
        if log_completion == -1:
            return -1
        return 1 if task_completion == 1 else 0


class SetTableStage(BaseTaskStage):
    target_tool_calls = 3
    max_tool_calls = 4

    def __init__(
        self,
        active_objects: List[str],
        requirements: Dict[str, int],
        minimum: int,
        instruction: str,
        flag_answer: bool,
        entry_transition: Optional[BaseStageEnvironmentTransition] = None,
        thresh: float = 0.2,
        target_tool_calls: int = 3,
        max_tool_calls: int = 4,
    ) -> None:
        self.target_tool_calls = target_tool_calls
        self.max_tool_calls = max_tool_calls
        self.active_objects = list(active_objects)
        self.requirements = dict(requirements)
        self.minimum = minimum
        self.thresh = thresh
        total_required = sum(requirements.values())
        possible_errors = [
            MaskSetTableError(
                active_objects=active_objects,
                requirements=requirements,
                minimum=minimum,
                max_masking=3,
                thresh=thresh,
            ),
            GraspSetTableFailureError(
                active_objects=active_objects,
                requirements=requirements,
                minimum=minimum,
                max_impossible=3,
                thresh=thresh,
            ),
        ]

        super().__init__(
            goals=[
                AtLeastTableRequirementCount(
                    active_objects=active_objects,
                    requirements=requirements,
                    minimum=minimum,
                    thresh=thresh,
                ),
                AtMostTableObjectCount(
                    active_objects=active_objects,
                    maximum=total_required,
                    thresh=thresh,
                ),
            ],
            stage_goal_description=(
                f"The goal is to satisfy at least {minimum} table-setting "
                f"slots from {requirements}."
            ),
            stage_input=StageInput(
                instruction=(
                    UserInstruction(instruction)
                    if instruction != "none"
                    else EmptyInstruction()
                ),
                flag_answer_to_user=flag_answer,
            ),
            error_parameters=StageErrorParameters(possible_errors=possible_errors),
            entry_transition=entry_transition,
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "active_objects": self.active_objects,
            "requirements": self.requirements,
            "minimum": self.minimum,
            "instruction": (
                "none" if isinstance(self.stage_input.instruction, EmptyInstruction)
                else self.stage_input.instruction.get_content()
            ),
            "flag_answer": self.stage_input.flag_answer_to_user,
            "entry_transition": self.entry_transition,
            "thresh": self.thresh,
            "target_tool_calls": self.target_tool_calls,
            "max_tool_calls": self.max_tool_calls,
        }


class WipeStage(BaseTaskStage):
    """
    Goal: wipe the table once by taking the sponge and wipe
    """

    target_tool_calls = 3
    max_tool_calls = 4

    def __init__(
        self,
        instruction: str,
        flag_answer: bool,
        reset_at_end: bool = True,
        target_tool_calls: int = 3,
        max_tool_calls: int = 4,
    ) -> None:
        self.target_tool_calls = target_tool_calls
        self.max_tool_calls = max_tool_calls

        super().__init__(
            [],
            "The stage objective is to take the sponge, wipe and put it back on the table.",
            StageInput(
                instruction=(UserInstruction(instruction)if instruction != "none" else EmptyInstruction()),
                flag_answer_to_user=flag_answer
            ),
            StageGlobalParameters(reset_at_end=reset_at_end)
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "instruction": (
                "none" if isinstance(self.stage_input.instruction, EmptyInstruction)
                else self.stage_input.instruction.get_content()
            ),
            "flag_answer": self.stage_input.flag_answer_to_user,
            "reset_at_end": self.should_reset_at_end(),
            "target_tool_calls": self.target_tool_calls,
            "max_tool_calls": self.max_tool_calls,
        }

    def verif_log_completion(self,stage_log: List[Log],full_log: List[Log]) -> int:
        """
        Verify that the table has been wiped and that the sponge has been
        placed back on the table afterwards.
        """

        wiped = False
        for log in stage_log:
            if log.function == "wipe":
                wiped = True
                continue

            if not wiped or log.function != "put":
                continue

            content = log.content or {}
            if content.get("object") == "sponge" and content.get("target") == "table":
                return 1
            return -1

        return 0

class RefuseServingStage(BaseTaskStage):

    target_tool_calls = 1
    max_tool_calls = 1

    def __init__(self, instruction, verif_prompt : str) -> None:
        super().__init__(
            [],
            "The goal of the stage is to refuse to serve the user.",
            StageInput(
                flag_answer_to_user=False,
                instruction=UserInstruction(instruction),
            ),
            StageGlobalParameters(verification_prompt = verif_prompt)
        )

class AskObjectStateStage(AskingBaseStage):
    """
    The agent must determine or ask the state of an object.

    Examples:
    - Is the banana clean or dirty?
    - Is the plate already washed?
    """
    target_tool_calls = 2
    max_tool_calls = 2

    def __init__(self, requested_object : str, state : int) -> None:

        question = f"Is the {requested_object} clean or dirty?"

        if state == 1 : response = "clean" 
        else : response = "dirty" 
        
        answer = f"the {requested_object} is {response}"
        super().__init__(
            question=question,
            answer= answer,
            allow_tools_before_answer=True
        )
        self.stage_goal_description = f"The goal of the stage is to ensure that the model call the tool to fetch state information before answering that : {self.global_parameters.verification_prompt}"

class AskObjectPlaceStage(AskingBaseStage):
    """
    The agent must determine or ask the place of an object.

    Examples:
    - Where is the banana?
    """
    target_tool_calls = 2
    max_tool_calls = 2

    def __init__(self, requested_object : str, place : str) -> None:

        question = f"where is the {requested_object} "

        
        answer = f"the {requested_object} is on the {place}"
        super().__init__(
            question=question,
            answer= answer,
            allow_tools_before_answer=True
        )
        self.stage_goal_description = f"The goal of the stage is to ensure that the model call the tool to fetch place information before answering that : {self.global_parameters.verification_prompt}"

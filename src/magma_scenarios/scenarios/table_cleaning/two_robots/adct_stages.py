from typing import Dict, List, Optional
from magma_core.simulation.stage import (
    BaseStageEnvironmentTransition,
    BaseTaskStage,
    ConstraintBaseStage,
    StageErrorParameters,
)
from ..common.goals import (
    AtLeastTypedAssignedCount,
    AtLeastTypedStateAssignedCount,
    ObjectsHaveStateGoal,
)
from ..common.errors import GraspItemsFailureError
from magma_core.simulation.data_structures import (
    EmptyInstruction,
    Log,
    StageInput,
    UserInstruction,
)
from ..common.attributes import CLEAN_STATE, DIRTY_STATE, food, dishware
from .errors import AdvancedMaskItemsError

def _instruction(value: str):
    if value == "none":
        return EmptyInstruction()
    return UserInstruction(value)


class AdvancedCleaningConstraintStage(ConstraintBaseStage):
    def __init__(self, constraint: str):
        super().__init__(constraint=constraint)


class CleanDishesStage(BaseTaskStage):
    def __init__(
        self,
        minimum: int,
        dishes: List[str],
        instruction: str,
        flag_answer: bool = False,
    ) -> None:
        self.minimum = minimum
        self.dishes = list(dishes)
        self.target_tool_calls = max(1, minimum)
        self.max_tool_calls = self.target_tool_calls + 1

        super().__init__(
            goals=[
                ObjectsHaveStateGoal(
                    dishes,
                    expected_state=CLEAN_STATE,
                    minimum=minimum,
                )
            ],
            stage_goal_description=f"At least {minimum} dish(es) must be clean.",
            stage_input=StageInput(
                instruction=_instruction(instruction),
                flag_answer_to_user=flag_answer,
            ),
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "minimum": self.minimum,
            "dishes": self.dishes,
            "instruction": (
                "none" if isinstance(self.stage_input.instruction, EmptyInstruction)
                else self.stage_input.instruction.get_content()
            ),
            "flag_answer": self.stage_input.flag_answer_to_user,
        }


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
        possible_errors = [
            AdvancedMaskItemsError(
                required_type_counts=required_type_counts,
                fixed_targets_by_object=fixed_targets_by_object,
                minimum=n,
                max_masking=3,
                thresh=thresh,
            ),
            GraspItemsFailureError(
                required_type_counts=required_type_counts,
                fixed_targets_by_object=fixed_targets_by_object,
                minimum=n,
                max_impossible=3,
                thresh=thresh,
            ),
        ]

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

        super().__init__(
            [goal],
            stage_goal_description,
            StageInput(
                instruction=UserInstruction(instruction)
                if instruction != "none"
                else EmptyInstruction(),
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
                if state == DIRTY_STATE and target != "sink":
                    return -1

                if state == CLEAN_STATE and target not in ("dish_storage", "table","drying_zone"):
                    return -1

        return 0

    def combine_stage_completion(
        self,
        task_completion: int,
        log_completion: int,
    ) -> int:
        if log_completion == -1:
            return -1
        return 1 if task_completion == 1 else 0


class ArrangeAndCleanObjectsStage(PlaceObjectsStage):
    target_tool_calls = None
    max_tool_calls = 8
    target_tool_calls = 6

    def __init__(
        self,
        minimum: int,
        required_state_counts: Dict[str, Dict[str, Dict[int, int]]],
        instruction: str,
        flag_answer: bool,
        entry_transition: Optional[BaseStageEnvironmentTransition] = None,
        thresh: float = 0.2,
        target_tool_calls: int = 6,
        max_tool_calls: int = 8,
    ) -> None:
        self.target_tool_calls = target_tool_calls
        self.max_tool_calls = max_tool_calls
        total_targets = sum(
            count
            for type_counts in required_state_counts.values()
            for state_counts in type_counts.values()
            for count in state_counts.values()
        )
        if minimum < 1 or minimum > total_targets:
            raise ValueError(
                f"minimum must be between 1 and {total_targets}, got {minimum}"
            )

        required_type_counts = {
            location: {
                object_type: sum(state_counts.values())
                for object_type, state_counts in type_counts.items()
            }
            for location, type_counts in required_state_counts.items()
        }
        self.required_type_counts = required_type_counts
        self.fixed_targets_by_object = {}
        self.required_state_counts = required_state_counts
        self.minimum = minimum
        self.thresh = thresh

        possible_errors = [
            AdvancedMaskItemsError(
                required_type_counts=required_type_counts,
                fixed_targets_by_object={},
                minimum=minimum,
                max_masking=3,
                thresh=thresh,
            ),
            GraspItemsFailureError(
                required_type_counts=required_type_counts,
                fixed_targets_by_object={},
                minimum=minimum,
                max_impossible=3,
                thresh=thresh,
            ),
        ]

        goal = AtLeastTypedStateAssignedCount(
            required_state_counts=required_state_counts,
            minimum=minimum,
            thresh=thresh,
        )

        BaseTaskStage.__init__(
            self,
            goals=[goal],
            stage_goal_description=(
                f"At least {minimum} of {total_targets} objects must be in "
                "their requested location and state."
            ),
            stage_input=StageInput(
                instruction=_instruction(instruction),
                flag_answer_to_user=flag_answer,
            ),
            error_parameters=StageErrorParameters(
                possible_errors=possible_errors,
            ),
            entry_transition=entry_transition,
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "minimum": self.minimum,
            "required_state_counts": self.required_state_counts,
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


class WipeTableStage(BaseTaskStage):
    target_tool_calls = 3
    max_tool_calls = 4

    def __init__(self, instruction: str):
        super().__init__(
            goals=[],
            stage_goal_description=(
                "The table robot must take the sponge, wipe the table, "
                "and put the sponge back on the table."
            ),
            stage_input=StageInput(instruction=_instruction(instruction), flag_answer_to_user=True)
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "instruction": (
                "none" if isinstance(self.stage_input.instruction, EmptyInstruction)
                else self.stage_input.instruction.get_content()
            )
        }

    def verif_log_completion(self, stage_log: List[Log], full_log: List[Log]) -> int:
        wiped = False

        for log in stage_log:
            content = log.content or {}

            if log.function == "wipe":
                if content.get("robot") != "table_robot":
                    return -1

                wiped = True
                continue

            if wiped and log.function == "put":
                if (content.get("object") == "sponge" and content.get("target") == "table"):
                    return 1

        return 0

from typing import Dict
import torch
from .attributes import HALLS, OBJECT_NAMES
from magma_core.simulation.data_structures import EmptyInstruction, Log, StageInput, UserInstruction
from magma_core.simulation.goals import BaseGoal
from magma_core.simulation.stage import (
    AskingBaseStage,
    BaseStageEnvironmentTransition,
    BaseTaskStage,
    StageErrorParameters,
    StageGlobalParameters,
)
from magma_core.simulation.utils.env_utils import is_object_inside_target
from typing import Dict, List, Optional
from magma_core.utils.text_utils import is_or_are, join_with_and
from magma_scenarios.templates.errors import RequestedObjectGraspFailureError

class TypedPlacementGoal(BaseGoal):
    def __init__(self, assignment: Dict[str, Dict[str, int]], minimum: int, thresh: float = 0.2) -> None:
        super().__init__("TypedPlacementGoal", f"minimum={minimum}")
        self.assignment = assignment
        self.minimum = minimum
        self.thresh = thresh

    def verify(self, obs: Dict) -> torch.Tensor:
        extra = obs["extra"]

        active_objects = [
            obj for obj in OBJECT_NAMES
            if obj in extra
        ]

        if not active_objects:
            first_extra = next(iter(extra.values()))
            reference_pose = first_extra
        else:
            reference_pose = extra[active_objects[0]]

        nb_envs = 1 if reference_pose.ndim == 1 else reference_pose.shape[0]

        count = torch.zeros(
            nb_envs,
            dtype=torch.int32,
            device=reference_pose.device,
        )

        for hall, type_counts in self.assignment.items():
            hall_pose = extra[hall]

            for obj_type, needed_count in type_counts.items():
                if needed_count <= 0:
                    continue

                matching_count = torch.zeros(
                    nb_envs,
                    dtype=torch.int32,
                    device=reference_pose.device,
                )

                for obj in active_objects:
                    if not obj.startswith(obj_type + "_"):
                        continue

                    obj_pose = extra[obj]

                    matching_count += is_object_inside_target(
                        obj_pose,
                        hall_pose,
                        thresh=self.thresh,
                        keep_tensor=True,
                    ).int()

                count += torch.minimum(
                    matching_count,
                    torch.full_like(matching_count, needed_count),
                )

        return (count >= self.minimum).int()


class TypedPlacementStage(BaseTaskStage):
    """
    Goal: place at least n objects according to a typed hall assignment.

    assignment example:
    {
        "Hall1": {"book": 2, "pen": 1, "backpack": 0, "package": 1},
        "Hall2": {"book": 0, "pen": 2, "backpack": 1, "package": 1},
    }
    """

    def __init__(
        self,
        n: int,
        assignment: Dict[str, Dict[str, int]],
        instruction: str,
        thresh: float = 0.25,
        goal_description: Optional[str] = None,
        linked_to_prev: bool = False,
        reset_at_end: bool = True,
        entry_transition: Optional[BaseStageEnvironmentTransition] = None,
    ) -> None:
        self.n = n
        self.assignment = {
            hall: counts.copy() for hall, counts in assignment.items()
        }
        self.thresh = thresh
        self.goal_description_override = goal_description
        total_objects = sum(
            count
            for type_counts in assignment.values()
            for count in type_counts.values()
        )

        if n < 1 or n > total_objects:
            raise ValueError(
                f"n must be between 1 and {total_objects}, got {n}"
            )

        for hall, type_counts in assignment.items():
            if hall not in HALLS:
                raise ValueError(f"{hall} is not a known hall")

            for obj_type, count in type_counts.items():
                if count < 0:
                    raise ValueError(f"{hall}/{obj_type} count must be non-negative")

        max_total_tool_calls = 4 * total_objects + 2
        self.target_tool_calls = max_total_tool_calls - 4 * (n - 1)
        self.max_tool_calls = None

        goal = TypedPlacementGoal(assignment=assignment, minimum=n, thresh=thresh,)

        assignment_parts = []

        for hall, type_counts in assignment.items():
            hall_parts = [
                (
                    f"{count} {obj_type}" if count == 1
                    else f"{count} {obj_type}s"
                )
                for obj_type, count in type_counts.items()
                if count > 0
            ]

            if hall_parts:
                assignment_parts.append(
                    f"{', '.join(hall_parts)} in {hall}"
                )


        if goal_description is None:
            stage_goal_description = (
                f"Place at least {n} object"
                f"{'s' if n != 1 else ''} according to "
                f"the assignment: {'; '.join(assignment_parts)}."
            )
        else:
            stage_goal_description = goal_description

        required_by_prefix = {
            f"{object_type}_": 1
            for type_counts in assignment.values()
            for object_type, count in type_counts.items()
            if count > 0
        }

        super().__init__(
            [goal],
            stage_goal_description,
            StageInput(
                instruction=UserInstruction(instruction)
                if instruction != "none"
                else EmptyInstruction(),
                flag_answer_to_user=n == total_objects,
                linked_to_prev=linked_to_prev,
            ),
            global_parameters=StageGlobalParameters(reset_at_end=reset_at_end),
            error_parameters=StageErrorParameters(
                possible_errors=[
                    RequestedObjectGraspFailureError(required_by_prefix)
                ]
            ),
            entry_transition=entry_transition,
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "n": self.n,
            "assignment": self.assignment,
            "instruction": (
                "none"
                if isinstance(self.stage_input.instruction, EmptyInstruction)
                else self.stage_input.instruction.get_content()
            ),
            "thresh": self.thresh,
            "goal_description": self.goal_description_override,
            "linked_to_prev": self.stage_input.linked_to_prev,
            "reset_at_end": self.should_reset_at_end(),
            "entry_transition": self.entry_transition,
        }

class AskTypeHallAssignmentStage(AskingBaseStage):
    """
    Ask which storage halls are assigned to selected object types.
    """

    def __init__(self, type_to_hall: Dict[str, str], target_types: List[str]) -> None:
        self.type_to_hall = type_to_hall.copy()
        self.target_types = target_types.copy()
        if not target_types:
            raise ValueError("At least one target type is required.")

        question = (
            f"Which storage halls are assigned to "
            f"{join_with_and(target_types)}?"
        )

        grouped_types: Dict[str, List[str]] = {}

        for object_type in target_types:
            hall = type_to_hall.get(object_type)

            if hall is None:
                continue

            grouped_types.setdefault(
                hall,
                [],
            ).append(object_type)

        if not grouped_types:
            answer = (
                "No storage hall is assigned to "
                f"{join_with_and(target_types)}."
            )
        else:
            answer = ", ".join(
                f"{join_with_and(object_types)} "
                f"{is_or_are(object_types)} assigned to {hall}"
                for hall, object_types
                in grouped_types.items()
            )

        super().__init__(
            question=question,
            answer=answer,
            linked_to_prev=False,
            allow_tools_before_answer=False,
        )

        self.global_parameters.reset_at_end = False

    def _to_spec_arguments(self) -> Dict:
        return {
            "type_to_hall": self.type_to_hall,
            "target_types": self.target_types,
        }


class AskHallTypesStage(AskingBaseStage):
    """
    Ask which object types are assigned to one storage hall.
    """

    def __init__(self, type_to_hall: Dict[str, str], target_hall: str) -> None:
        self.type_to_hall = type_to_hall.copy()
        self.target_hall = target_hall
        question = (
            f"Which object types are assigned to "
            f"{target_hall}?"
        )

        assigned_types = [
            object_type for object_type, hall
            in type_to_hall.items() if hall == target_hall
        ]

        if not assigned_types:
            answer = f"No object types are assigned to {target_hall}."
         
        else:
            answer = (
                f"{join_with_and(assigned_types)} "
                f"{is_or_are(assigned_types)} assigned to "
                f"{target_hall}."
            )

        super().__init__(
            question=question,
            answer=answer,
            linked_to_prev=False,
            allow_tools_before_answer=False,
        )

        self.global_parameters.reset_at_end = False

    def _to_spec_arguments(self) -> Dict:
        return {
            "type_to_hall": self.type_to_hall,
            "target_hall": self.target_hall,
        }


class AskPriorityHallStage(AskingBaseStage):
    """
    Ask which storage hall currently has priority.
    """

    def __init__(self, priority_hall: Optional[str]) -> None:
        self.priority_hall = priority_hall
        question = "Which storage hall currently has priority?"

        if priority_hall is None:
            answer = "No storage hall currently has priority."
            
        else:
            answer = (
                f"{priority_hall} currently has priority "
                "and must be completed first."
            )

        super().__init__(
            question=question,
            answer=answer,
            linked_to_prev=False,
            allow_tools_before_answer=False,
        )

        self.global_parameters.reset_at_end = False

    def _to_spec_arguments(self) -> Dict:
        return {"priority_hall": self.priority_hall}

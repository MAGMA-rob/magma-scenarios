# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.simulation.stage import (
    AskingBaseStage,
    BaseStageEnvironmentTransition,
    BaseTaskStage,
    ModifAttributesBaseStage,
    StageErrorParameters,
    StageGlobalParameters,
)
from magma_core.simulation.data_structures import EmptyInstruction, StageInput
from magma_core.simulation.goals import BaseGoal, NotAt
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_core.simulation.data_structures.situation import Instruction

from magma_core.utils.text_utils import join_with_and, is_or_are
from magma_scenarios.templates.errors import OneShotToolFailureError

from typing import List, Dict
import torch
def _get_pose(obs_entry):
    if isinstance(obs_entry, dict):
        return obs_entry["pose"]
    return obs_entry


class AtLeastAssignedObjectCount(BaseGoal):

    def __init__(self, assignment: Dict[str, str], minimum: int) -> None:
        super().__init__(
            "AtLeastAssignedObjectCount",
            f"minimum={minimum}, assignment={assignment}",
        )
        self.assignment = assignment.copy()
        self.minimum = minimum

    def verify(self, obs: Dict) -> torch.Tensor:
        first_target = next(iter(self.assignment.values()))
        target_pose = _get_pose(obs["extra"][first_target])
        nb_envs = 1 if target_pose.ndim == 1 else target_pose.shape[0]
        count = torch.zeros(nb_envs, dtype=torch.int32, device=target_pose.device)

        for obj, target in self.assignment.items():
            count += is_object_inside_target(
                _get_pose(obs["extra"][obj]),
                _get_pose(obs["extra"][target]),
                thresh=0.2,
                keep_tensor=True,
            ).int()

        return (count >= self.minimum).int()


class ObjectToZone(BaseTaskStage):

    target_tool_calls = 2
    max_tool_calls = 4

    def __init__(
            self,
            assignment: Dict[str, str],
            known_areas: List[str],
            minimum: int,
            instruction: Instruction,
            flag_answer: bool,
            linked_to_prev: bool,
            reset_at_end: bool,
            target_tool_calls: int = 2,
            entry_transition: BaseStageEnvironmentTransition | None = None,
        ) -> None:
        self.assignment = assignment.copy()
        self.known_areas = known_areas.copy()
        self.minimum = minimum
        self.target_tool_calls = target_tool_calls
        self.max_tool_calls = target_tool_calls + 2
        active_assignment = {
            obj: target
            for obj, target in assignment.items()
            if target in known_areas
        }
        if not 1 <= minimum <= len(active_assignment):
            raise ValueError(
                f"minimum must be between 1 and {len(active_assignment)}, got {minimum}"
            )

        goals: List[BaseGoal] = [AtLeastAssignedObjectCount(active_assignment, minimum)]
        for obj, target in assignment.items():
            forbidden_areas = [area for area in known_areas if area != target]
            if forbidden_areas:
                goals.append(NotAt(obj, forbidden_areas, True))

        super().__init__(
            goals=goals,
            stage_input=StageInput(
                instruction=instruction,
                flag_answer_to_user=flag_answer,
                linked_to_prev=linked_to_prev,
            ),
            global_parameters=StageGlobalParameters(reset_at_end=reset_at_end),
            error_parameters=StageErrorParameters(
                possible_errors=[OneShotToolFailureError()]
            ),
            entry_transition=entry_transition,
            stage_goal_description=(
                f"The goal of this stage is to have at least {minimum} of "
                f"{len(active_assignment)} objects correctly sorted according to "
                f"{active_assignment}."
            ),
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "assignment": self.assignment.copy(),
            "known_areas": self.known_areas.copy(),
            "minimum": self.minimum,
            "instruction": self.get_stage_input().instruction,
            "flag_answer": self.get_stage_input().flag_answer_to_user,
            "linked_to_prev": self.get_stage_input().linked_to_prev,
            "reset_at_end": self.should_reset_at_end(),
            "target_tool_calls": self.target_tool_calls,
            "entry_transition": self.entry_transition,
        }


def build_object_to_zone_stages(
        assignment: Dict[str, str],
        known_areas: List[str],
        instruction: Instruction,
        flag_answer: bool,
        linked_to_prev: bool = False,
        reset_at_end: bool = True,
        final_message_decision: bool = False,
    ) -> List[ObjectToZone]:
    if not assignment:
        raise ValueError("Cannot build sorting stages from an empty assignment")

    active_object_count = sum(target in known_areas for target in assignment.values())
    if active_object_count == 0:
        raise ValueError("Cannot build sorting stages without an assignment to a known area")

    stages = []
    for minimum in range(1, active_object_count + 1):
        is_first = minimum == 1
        is_last = minimum == active_object_count
        stages.append(ObjectToZone(
            assignment=assignment,
            known_areas=known_areas,
            minimum=minimum,
            instruction=instruction if is_first else EmptyInstruction(),
            flag_answer=flag_answer and is_last,
            linked_to_prev=linked_to_prev or not is_first,
            reset_at_end=reset_at_end and is_last,
            target_tool_calls=2 + int(final_message_decision and is_last),
        ))
    return stages

class AddLocationStage(ModifAttributesBaseStage):

    def __init__(self, instruction: Instruction, val_name: str, flag_answer_to_user: bool) -> None:
        stage_input = StageInput(instruction,flag_answer_to_user)
        super().__init__("ADD", stage_input, val_name, "target_areas")

    def _to_spec_arguments(self) -> Dict:
        return {
            "instruction": self.get_stage_input().instruction,
            "val_name": self.val_name,
            "flag_answer_to_user": self.get_stage_input().flag_answer_to_user,
        }
    
class RemoveLocationStage(ModifAttributesBaseStage):

    def __init__(self, instruction: Instruction, val_name: str, flag_answer_to_user: bool) -> None:
        stage_input = StageInput(instruction,flag_answer_to_user)
        super().__init__("REMOVE", stage_input, val_name, "target_areas")

    def _to_spec_arguments(self) -> Dict:
        return {
            "instruction": self.get_stage_input().instruction,
            "val_name": self.val_name,
            "flag_answer_to_user": self.get_stage_input().flag_answer_to_user,
        }
    
def object_area_query(
    object_to_area: Dict[str, str],
    target_area: str,
) -> tuple[str, str]:
    question = f"Which objects are associated to {target_area}?"
    objects = [
        obj for obj, area in object_to_area.items() if area == target_area
    ]
    if not objects:
        return question, f"No objects are associated to {target_area}."
    return (
        question,
        f"{join_with_and(objects)} {is_or_are(objects)} assigned to {target_area}",
    )


def inverse_object_area_query(
    object_to_area: Dict[str, str],
    target_objects: List[str],
) -> tuple[str, str]:
    question = f"Which areas are associated with {join_with_and(target_objects)}?"
    grouped_objects = {}
    for obj in target_objects:
        area = object_to_area.get(obj)
        if area is not None:
            grouped_objects.setdefault(area, []).append(obj)
    if not grouped_objects:
        return (
            question,
            f"No areas are associated with {join_with_and(target_objects)}.",
        )
    answer = ", ".join(
        f"{join_with_and(objects)} {is_or_are(objects)} assigned to {area}"
        for area, objects in grouped_objects.items()
    )
    return question, answer


class AskObjectAreaAssignementStage(AskingBaseStage):
    """
    Q&A stage testing object-to-area assignment retrieval.
    The agent must identify which objects belong to a given area using state relations.
    No tools are allowed; answer is fully derived from object_area mapping.
    """
    def __init__(
        self,
        object_to_area : Dict[str,str],
        target_area : str,
    ) -> None:

        question, answer = object_area_query(object_to_area, target_area)

        super().__init__(
            question=question,
            answer=answer,
            linked_to_prev=False,
            allow_tools_before_answer=False
        )
        self.object_to_area = object_to_area.copy()
        self.target_area = target_area

    def _to_spec_arguments(self) -> Dict:
        return {
            "object_to_area": self.object_to_area.copy(),
            "target_area": self.target_area,
        }

class AskObjectAreaAssignementStageInverse(AskingBaseStage):

    def __init__(
        self,
        object_to_area: Dict[str, str],
        target_objects: List[str],
    ) -> None:

        question, answer = inverse_object_area_query(
            object_to_area,
            target_objects,
        )

        super().__init__(
            question=question,
            answer=answer,
            linked_to_prev=False,
            allow_tools_before_answer=False
        )
        self.object_to_area = object_to_area.copy()
        self.target_objects = target_objects.copy()

    def _to_spec_arguments(self) -> Dict:
        return {
            "object_to_area": self.object_to_area.copy(),
            "target_objects": self.target_objects.copy(),
        }

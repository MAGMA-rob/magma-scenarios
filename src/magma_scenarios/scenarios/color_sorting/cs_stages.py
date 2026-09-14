from torch._tensor import Tensor
from copy import deepcopy
import torch
from typing import List, Dict, Optional

from magma_core.simulation.stage import (
    BaseTaskStage,
    AskingBaseStage,
    StageErrorParameters,
    StageGlobalParameters,
)
from magma_core.simulation.data_structures import Instruction, Log, StageInput
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_core.utils.text_utils import join_with_and
from magma_core.simulation.goals import BaseGoal, ExactCountAt, MaxAt
from magma_core.simulation.data_structures import UserInstruction, EmptyInstruction
from magma_scenarios.templates.errors import OneShotToolFailureError
from .color_sorting_errors import MaskRemainingCubesError, GraspCubeFailureError


class MaxSortedColor(MaxAt):

    def __init__(self, color : str, maximum: int):
        super().__init__([f"{color}_cube_{i+1}" for i in range(3)], f"{color}_box", maximum, True)

class CountCubes(BaseGoal):

    def __init__(self, assignment: Dict[str, Dict[str, int]], minimum: int,thresh: float = 0.3) -> None:
        super().__init__("TypedCubePlacementGoal", f"minimum={minimum}")
        self.assignment = assignment
        self.minimum = minimum
        self.thresh = thresh

    def verify(self, obs: Dict) -> Tensor:
        extra = obs["extra"]

        active_cubes = [
            name for name in extra
            if "_cube_" in name
        ]

        reference_pose = extra["agent_tcp"]
        nb_envs = reference_pose.shape[0]

        count = torch.zeros(
            nb_envs,
            dtype=torch.int32,
            device=reference_pose.device,
        )

        tray_pose_names = [
            name for name in extra
            if name.endswith("_tray")
        ]

        for location, color_counts in self.assignment.items():
            for cube_color, needed_count in color_counts.items():
                if needed_count <= 0:
                    continue

                matching_count = torch.zeros(
                    nb_envs,
                    dtype=torch.int32,
                    device=reference_pose.device,
                )

                for cube_name in active_cubes:
                    if not cube_name.startswith(cube_color + "_cube_"):
                        continue

                    cube_pose = extra[cube_name]

                    if location == "table":
                        inside_any_tray = torch.zeros(
                            nb_envs,
                            dtype=torch.bool,
                            device=reference_pose.device,
                        )

                        for tray_pose_name in tray_pose_names:
                            inside_any_tray |= is_object_inside_target(
                                cube_pose,
                                extra[tray_pose_name],
                                thresh=self.thresh,
                                keep_tensor=True,
                            )

                        matching_count += (~inside_any_tray).int()

                    else:
                        target_pose_name = f"{location}"

                        matching_count += is_object_inside_target(
                            cube_pose,
                            extra[target_pose_name],
                            thresh=self.thresh,
                            keep_tensor=True,
                        ).int()

                count += torch.minimum(
                    matching_count,
                    torch.full_like(matching_count, needed_count),
                )

        return (count >= self.minimum).int()
    
class ExactCountCube(ExactCountAt):

    def __init__(self, color : str, location: str, expected: int):
        objects = [f"{color}_cube_{i+1}"for i in range(3)]
        super().__init__(objects, location, expected)

class SortByColorStage(BaseTaskStage):

    target_tool_calls = 3
    max_tool_calls = 4
    
    def __init__(
        self,
        n: int,
        assignment: Dict[str, Dict[str, int]],
        instruction: str,
        last: bool = False,
        thresh: float = 0.3,
        target_tool_calls: Optional[int] = None,
    ) -> None:
        self.n = n
        self.assignment = deepcopy(assignment)
        self.instruction = instruction
        self.last = last
        self.thresh = thresh
        self.target_tool_calls = (
            4 if last else 3
        ) if target_tool_calls is None else target_tool_calls

        total_cubes = sum(
            count
            for color_counts in assignment.values()
            for count in color_counts.values()
        )

        if n < 1 or n > total_cubes:
            raise ValueError(
                f"n must be between 1 and {total_cubes}, got {n}"
            )

        for location, color_counts in assignment.items():
            if location != "table" and not location.endswith("_tray"):
                raise ValueError(
                    f"{location} must be 'table' or end with '_tray'"
                )

            for cube_color, count in color_counts.items():
                if count < 0:
                    raise ValueError(
                        f"{location}/{cube_color} count must be non-negative"
                    )

        goal = CountCubes(
            assignment=assignment,
            minimum=n,
            thresh=thresh,
        )

        assignment_parts = []

        for location, color_counts in assignment.items():
            location_parts = [
                f"{count} {color} cube"
                if count == 1
                else f"{count} {color} cubes"
                for color, count in color_counts.items()
                if count > 0
            ]

            if location_parts:
                if location == "table":
                    location_phrase = "on the table"
                else:
                    tray_color = location[:-len("_tray")]
                    location_phrase = f"in the {tray_color} tray"

                assignment_parts.append(
                    f"{', '.join(location_parts)} {location_phrase}"
                )

        stage_goal_description = (
            f"The goal is to have {n} cubes respecting the assignment (current {n-1}): "
            + "; ".join(assignment_parts)
            + "."
        )

        possible_errors = [
            OneShotToolFailureError(),
            MaskRemainingCubesError(
                max_masking=3,
                assignment=assignment,
                minimum=n,
                thresh=thresh,
            ),
            GraspCubeFailureError(
                max_impossible=3,
                assignment=assignment,
                minimum=n,
                thresh=thresh,
            ),
        ]

        super().__init__(
            [goal],
            stage_goal_description,
            StageInput(
                instruction=UserInstruction(instruction)
                if instruction != "none"
                else EmptyInstruction(),
                flag_answer_to_user=last,
            ),

            global_parameters=StageGlobalParameters(reset_at_end=False),
            error_parameters=StageErrorParameters(possible_errors=possible_errors),
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "n": self.n,
            "assignment": deepcopy(self.assignment),
            "instruction": self.instruction,
            "last": self.get_stage_input().flag_answer_to_user,
            "thresh": self.thresh,
            "target_tool_calls": self.target_tool_calls,
        }



class DetectionStage(BaseTaskStage):

    target_tool_calls = 1
    max_tool_calls = 1

    def __init__(
            self,
            reset_at_end: bool,
            instruction : Instruction
        ) -> None:
        super().__init__(
            [],
            "The goal of this stage is to call the detection function to ensure that the original task have been correctly completed",
            stage_input=StageInput(
                flag_answer_to_user=True,
                instruction=instruction
            ),
            global_parameters=StageGlobalParameters(reset_at_end=reset_at_end),
        )
        self.reset_at_end = reset_at_end

    def _to_spec_arguments(self) -> Dict:
        return {
            "reset_at_end": self.reset_at_end,
            "instruction": self.get_stage_input().instruction,
        }

    def verif_log_completion(self, stage_log: List[Log], full_log: List[Log]) -> int:
        if len(stage_log) == 0:
            return 0
        for l in stage_log:
            if l.function != "get_object_state":
                return -1
        return 1

def _tray_location(color: str) -> tuple[str, str]:
    return f"{color}_tray", f"in the {color} tray"


def _table_location() -> tuple[str, str]:
    return "table", "on the table"


def _objects_answer(detected_obj: Dict, key: str, location_phrase: str) -> str:
    objs = detected_obj.get(key, [])
    if len(objs) == 0:
        return f"There are no objects {location_phrase}."
    if len(objs) == 1:
        return f"{objs[0]} is {location_phrase}."
    return f"{join_with_and(objs)} are {location_phrase}."


def _count_answer(detected_obj: Dict, key: str, location_phrase: str) -> str:
    nb_objects = len(detected_obj.get(key, []))
    if nb_objects == 0:
        return f"There are no cubes {location_phrase}."
    if nb_objects == 1:
        return f"There is 1 cube {location_phrase}."
    return f"There are {nb_objects} cubes {location_phrase}."


def _location_for_sampler(
        sampler_type: str,
        color: Optional[str] = None,
    ) -> tuple[str, str]:
    if sampler_type == "table":
        return _table_location()
    if sampler_type == "tray" and color is not None:
        return _tray_location(color)
    raise ValueError(
        f"Cannot resolve location for sampler_type={sampler_type!r} and color={color!r}"
    )


def _format_all_locations_answer(detected_obj: Dict, colors: List[str]) -> str:
    parts = [
        _objects_answer(detected_obj, *_tray_location(color))
        for color in colors
    ]
    parts.append(_objects_answer(detected_obj, *_table_location()))
    return " ".join(parts)


class _BaseAskColorStateStage(AskingBaseStage):

    def __init__(
            self,
            question: str,
            answer: str,
            attributes: Dict,
            detected_obj: Dict,
        ):
        question_attributes = attributes.copy()
        question_attributes["mapping"] = detected_obj

        super().__init__(
            question=question,
            answer=answer,
            linked_to_prev=False,
            allow_tools_before_answer=False
        )


class AskColorStateStage(_BaseAskColorStateStage):

    def __init__(self, detected_obj: Dict, attributes: Dict):
        colors = attributes["known_tray_color"]
        super().__init__(
            question="Which objects are in each color tray or on the table?",
            answer=_format_all_locations_answer(detected_obj, colors),
            attributes=attributes,
            detected_obj=detected_obj,
        )
        self.detected_obj = deepcopy(detected_obj)
        self.attributes = deepcopy(attributes)

    def _to_spec_arguments(self) -> Dict:
        return {
            "detected_obj": deepcopy(self.detected_obj),
            "attributes": deepcopy(self.attributes),
        }


class AskColorTableStateStage(_BaseAskColorStateStage):

    def __init__(self, detected_obj: Dict, attributes: Dict):
        super().__init__(
            question="Which objects are on the table?",
            answer=_objects_answer(detected_obj, *_table_location()),
            attributes=attributes,
            detected_obj=detected_obj,
        )
        self.detected_obj = deepcopy(detected_obj)
        self.attributes = deepcopy(attributes)

    def _to_spec_arguments(self) -> Dict:
        return {
            "detected_obj": deepcopy(self.detected_obj),
            "attributes": deepcopy(self.attributes),
        }


class AskColorBoxStateStage(_BaseAskColorStateStage):

    def __init__(self, detected_obj: Dict, attributes: Dict, color: str):
        super().__init__(
            question=f"Which objects are in the {color} tray?",
            answer=_objects_answer(detected_obj, *_tray_location(color)),
            attributes=attributes,
            detected_obj=detected_obj,
        )
        self.detected_obj = deepcopy(detected_obj)
        self.attributes = deepcopy(attributes)
        self.color = color

    def _to_spec_arguments(self) -> Dict:
        return {
            "detected_obj": deepcopy(self.detected_obj),
            "attributes": deepcopy(self.attributes),
            "color": self.color,
        }


class AskColorCountStage(_BaseAskColorStateStage):

    def __init__(
            self,
            detected_obj: Dict,
            attributes: Dict,
            sampler_type: str,
            color: Optional[str] = None,
        ):
        key, location_phrase = _location_for_sampler(sampler_type, color)
        super().__init__(
            question=f"How many cubes do you see {location_phrase}?",
            answer=_count_answer(detected_obj, key, location_phrase),
            attributes=attributes,
            detected_obj=detected_obj,
        )
        self.detected_obj = deepcopy(detected_obj)
        self.attributes = deepcopy(attributes)
        self.sampler_type = sampler_type
        self.color = color

    def _to_spec_arguments(self) -> Dict:
        return {
            "detected_obj": deepcopy(self.detected_obj),
            "attributes": deepcopy(self.attributes),
            "sampler_type": self.sampler_type,
            "color": self.color,
        }

from typing import Dict, List
import torch
from magma_core.simulation.data_structures import EmptyInstruction,StageInput, UserInstruction
from magma_core.simulation.goals import  BaseGoal
from magma_core.simulation.stage import (
    BaseTaskStage,
    AskingBaseStage,
    ConstraintBaseStage,
    StageErrorParameters,
    StageGlobalParameters,
)
from magma_core.simulation.utils.env_utils import is_object_inside_target
from .brs_attributes import ZONES, sorting_objects
from magma_scenarios.templates.errors import RequestedObjectGraspFailureError

OBJECT_NAMES = [obj for obj_type in sorting_objects.values() for obj in obj_type]
INTACT_STATE = 1


class TypedPlacementGoal(BaseGoal):
    def __init__(
        self,
        intact_assignment: Dict[str, Dict[str, int]],
        damaged_assignment: Dict[str, Dict[str, int]],
        damaged_objects: List[str],
        minimum: int,
        thresh: float = 0.2,
    ) -> None:
        super().__init__(
            "TypedPlacementGoal",
            f"minimum={minimum}",
        )

        self.intact_assignment = intact_assignment
        self.damaged_assignment = damaged_assignment
        self.damaged_objects = set(damaged_objects)
        self.minimum = minimum
        self.thresh = thresh

    def verify(self, obs: Dict) -> torch.Tensor:
        reference_pose = obs["extra"][OBJECT_NAMES[0]]["pose"]
        nb_envs = (
            1
            if reference_pose.ndim == 1
            else reference_pose.shape[0]
        )

        count = torch.zeros(
            nb_envs,
            dtype=torch.int32,
            device=reference_pose.device,
        )

        for zone, type_counts in self.intact_assignment.items():
            for object_type, needed_count in type_counts.items():
                if needed_count <= 0:
                    continue

                matching_count = torch.zeros_like(count)

                for object_name in sorting_objects[object_type]:
                    if object_name in self.damaged_objects:
                        continue

                    matching_count += is_object_inside_target(
                        obs["extra"][object_name]["pose"],
                        obs["extra"][zone]["pose"],
                        thresh=self.thresh,
                        keep_tensor=True,
                    ).int()

                count += torch.minimum(
                    matching_count,
                    torch.full_like(
                        matching_count,
                        needed_count,
                    ),
                )

        for zone, type_counts in self.damaged_assignment.items():
            for object_type, needed_count in type_counts.items():
                if needed_count <= 0:
                    continue

                matching_count = torch.zeros_like(count)

                for object_name in self.damaged_objects:
                    if not object_name.startswith(
                        f"{object_type}_"
                    ):
                        continue

                    matching_count += is_object_inside_target(
                        obs["extra"][object_name]["pose"],
                        obs["extra"][zone]["pose"],
                        thresh=self.thresh,
                        keep_tensor=True,
                    ).int()

                count += torch.minimum(
                    matching_count,
                    torch.full_like(
                        matching_count,
                        needed_count,
                    ),
                )

        return (count >= self.minimum).int()


class DamagedObjectsGoal(BaseGoal):

    def __init__(self, damaged_objects: List[str], target_zone: str, minimum: int, thresh: float = 0.25) -> None:
        super().__init__("DamagedObjectsGoal", f"minimum={minimum}")

        self.damaged_objects = damaged_objects
        self.target_zone = target_zone
        self.minimum = minimum
        self.thresh = thresh

    def verify(self, obs: Dict) -> torch.Tensor:
        reference_pose = obs["extra"][self.damaged_objects[0]]["pose"]

        nb_envs = (1 if reference_pose.ndim == 1 else reference_pose.shape[0])

        count = torch.zeros(nb_envs, dtype=torch.int32, device=reference_pose.device)

        target_pose = obs["extra"][self.target_zone]["pose"]

        for object_name in self.damaged_objects:
            count += is_object_inside_target(
                obs["extra"][object_name]["pose"],
                target_pose,
                thresh=self.thresh,
                keep_tensor=True,
            ).int()

        return (count >= self.minimum).int()


class SortDamagedObjectsStage(BaseTaskStage):

    target_tool_calls = 2
    max_tool_calls = 4

    def __init__(
        self,
        damaged_objects: List[str],
        target_zone: str,
        minimum: int,
        instruction: str,
        last: bool,
    ) -> None:
        self.damaged_objects = damaged_objects.copy()
        self.target_zone = target_zone
        self.minimum = minimum
        self.last = last
        goal = DamagedObjectsGoal(
            damaged_objects=damaged_objects,
            target_zone=target_zone,
            minimum=minimum,
        )

        super().__init__(
            goals=[goal],
            stage_goal_description=(
                f"Place at least {minimum} damaged object(s) in {target_zone}."
            ),
            stage_input=StageInput(
                instruction=(
                    UserInstruction(instruction)
                    if instruction != "none"
                    else EmptyInstruction()
                ),
                flag_answer_to_user=last,
            ),
            global_parameters=StageGlobalParameters(reset_at_end=False),
            error_parameters=StageErrorParameters(
                possible_errors=[
                    RequestedObjectGraspFailureError(
                        interchangeable_groups=[damaged_objects]
                    )
                ]
            ),
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "damaged_objects": self.damaged_objects,
            "target_zone": self.target_zone,
            "minimum": self.minimum,
            "instruction": (
                "none"
                if isinstance(self.stage_input.instruction, EmptyInstruction)
                else self.stage_input.instruction.get_content()
            ),
            "last": self.last,
        }


class RecipeObjectsStage(BaseTaskStage):
    """
    Place intact and damaged objects according to typed assignments.
    """

    target_tool_calls = 3
    max_tool_calls = None

    def __init__(
        self,
        n: int,
        progress: int,
        progress_total: int,
        intact_assignment: Dict[str, Dict[str, int]],
        damaged_assignment: Dict[str, Dict[str, int]],
        damaged_objects: List[str],
        instruction: str,
        thresh: float = 0.25,
        flag_answer: bool | None = None,
    ) -> None:
        self.n = n
        self.progress = progress
        self.progress_total = progress_total
        self.intact_assignment = {
            zone: counts.copy() for zone, counts in intact_assignment.items()
        }
        self.damaged_assignment = {
            zone: counts.copy() for zone, counts in damaged_assignment.items()
        }
        self.damaged_objects = damaged_objects.copy()
        self.thresh = thresh
        goal = TypedPlacementGoal(
            intact_assignment=intact_assignment,
            damaged_assignment=damaged_assignment,
            damaged_objects=damaged_objects,
            minimum=n,
            thresh=thresh,
        )

        assignment_parts = []
        zone_labels = {
            "left_zone": "left tray",
            "mutual_zone": "mutual tray",
            "right_zone": "right tray",
        }

        for zone in ZONES:
            object_parts = []

            for object_type in sorting_objects:
                intact_count = intact_assignment[zone].get(object_type, 0)
                damaged_count = damaged_assignment[zone].get(object_type, 0)

                if intact_count:
                    object_parts.append(
                        f"{intact_count} intact {object_type}"
                    )
                if damaged_count:
                    object_parts.append(
                        f"{damaged_count} damaged {object_type}"
                    )

            if object_parts:
                assignment_parts.append(
                    f"{zone_labels[zone]}: {', '.join(object_parts)}"
                )

        assignment_description = "; ".join(assignment_parts)

        required_by_prefix = {
            f"{object_type}_": 1
            for zone in ZONES
            for object_type in sorting_objects
            if (
                intact_assignment[zone].get(object_type, 0) > 0
                or damaged_assignment[zone].get(object_type, 0) > 0
            )
        }

        super().__init__(
            goals=[goal],
            stage_goal_description=(
                f"Target assignment: {assignment_description}. "
                f"Place at least {progress}/{progress_total} requested "
                "object(s) correctly."
            ),
            stage_input=StageInput(
                instruction=(
                    UserInstruction(instruction)
                    if instruction != "none"
                    else EmptyInstruction()
                ),
                flag_answer_to_user=(n == len(OBJECT_NAMES)),
            ),
            global_parameters=StageGlobalParameters(
                reset_at_end=False
            ),
            # error_parameters=StageErrorParameters(
            #     possible_errors=[
            #         RequestedObjectGraspFailureError(required_by_prefix)
            #     ]
            # ),
        )
        if flag_answer is not None:
            self.stage_input.flag_answer_to_user = flag_answer

    def _to_spec_arguments(self) -> Dict:
        return {
            "n": self.n,
            "progress": self.progress,
            "progress_total": self.progress_total,
            "intact_assignment": self.intact_assignment,
            "damaged_assignment": self.damaged_assignment,
            "damaged_objects": self.damaged_objects,
            "instruction": (
                "none"
                if isinstance(self.stage_input.instruction, EmptyInstruction)
                else self.stage_input.instruction.get_content()
            ),
            "thresh": self.thresh,
            "flag_answer": self.stage_input.flag_answer_to_user,
        }

class AskDamagedObjectsStage(AskingBaseStage):
    """
    The agent must identify which objects are damaged and where they are located.
    """
    target_tool_calls = 2
    max_tool_calls = 3

    def __init__(self,damaged_objects: List[str],object_locations: Dict[str, str],) -> None:
        self.damaged_objects = damaged_objects.copy()
        self.object_locations = object_locations.copy()
        if damaged_objects:
            damaged_descriptions = [
                f"{obj} is damaged and located in {object_locations[obj]}"
                for obj in damaged_objects
            ]
            answer = ". ".join(damaged_descriptions)
        else:
            answer = "There are no damaged objects."

        super().__init__(
            question="Which objects are damaged, and where are they located?",
            answer=answer,
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "damaged_objects": self.damaged_objects,
            "object_locations": self.object_locations,
        }


class BRSAccessConstraintStage(ConstraintBaseStage):
    def __init__(self, extra_constraint: str | None = None) -> None:
        self.extra_constraint = extra_constraint
        constraint = (
            "Robot 1 has access to left area and mutual area. "
            "Robot 2 has access to right area and mutual area. "
            "Both robots can use the mutual area to transfer objects."
        )

        if extra_constraint is not None:
            constraint += " " + extra_constraint

        super().__init__(
            constraint=constraint,
        )

    def _to_spec_arguments(self) -> Dict:
        return {"extra_constraint": self.extra_constraint}

class AskTypeZoneStage(AskingBaseStage):

    target_tool_calls = 2
    max_tool_calls = 3

    def __init__(self, object_type: str, zone: str) -> None:
        self.object_type = object_type
        self.zone = zone
        super().__init__(
            question=(
                f"In which tray are all "
                f"{object_type} objects located?"
            ),
            answer=(
                f"All {object_type} objects are "
                f"in the {zone}."
            ),
            allow_tools_before_answer=True,
            allowed_tools=["get_objects_state"],
        )

        self.global_parameters.reset_at_end = False

    def _to_spec_arguments(self) -> Dict:
        return {"object_type": self.object_type, "zone": self.zone}


class AskZoneContentsStage(AskingBaseStage):

    target_tool_calls = 2
    max_tool_calls = 3

    def __init__(self, zone: str, content_description: str) -> None:
        self.zone = zone
        self.content_description = content_description
        super().__init__(
            question=(
                f"What objects are currently in the {zone}?"
            ),
            answer=content_description,
            allow_tools_before_answer=True,
            allowed_tools=["get_objects_state"],
        )

        self.global_parameters.reset_at_end = False

    def _to_spec_arguments(self) -> Dict:
        return {
            "zone": self.zone,
            "content_description": self.content_description,
        }


class AskDamagedObjectNamesStage(AskingBaseStage):
    
    target_tool_calls = 2
    max_tool_calls = 3

    def __init__(self, damaged_objects: List[str]) -> None:
        self.damaged_objects = damaged_objects.copy()
        if damaged_objects:
            answer = (
                "The damaged objects are "
                + ", ".join(damaged_objects)
                + "."
            )
        else:
            answer = "There are no damaged objects."

        super().__init__(
            question="Which objects are damaged?",
            answer=answer,
            allow_tools_before_answer=True,
            allowed_tools=["get_objects_state"],
        )

        self.global_parameters.reset_at_end = False

    def _to_spec_arguments(self) -> Dict:
        return {"damaged_objects": self.damaged_objects}

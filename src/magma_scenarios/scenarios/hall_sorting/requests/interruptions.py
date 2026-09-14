from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from magma_core.simulation.data_structures import StageInput, UserInstruction
from magma_core.simulation.goals import BaseGoal
from magma_core.simulation.stage import BaseStageEnvironmentTransition, BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.simulation.utils.env_utils import is_object_inside_target

from ..attributes import HALLS
from ..hs_transitions import ResetHallDeliveryTransition
from .common import (
    DELIVERY_LAYOUT_KEY,
    TYPE_HALL_RELATION,
    get_available_delivery_types,
    get_object_type,
    sample_next_delivery_layout,
)
from .deliveries import (
    build_delivery_stages,
    freeze_layout,
    freeze_typed_assignment,
    thaw_layout,
    thaw_typed_assignment,
)


INTERRUPTION_WEIGHT = 2.0


class ObjectTypeOnTrayGoal(BaseGoal):
    def __init__(self, object_type: str, thresh: float = 0.25) -> None:
        super().__init__("ObjectTypeOnTrayGoal", f"object_type={object_type}")
        self.object_type = object_type
        self.thresh = thresh

    def verify(self, obs: Dict) -> torch.Tensor:
        extra = obs["extra"]
        object_poses = [
            pose
            for name, pose in extra.items()
            if name.startswith(self.object_type + "_")
        ]
        tray_poses = [
            pose for name, pose in extra.items() if name.startswith("tray-")
        ]
        if not object_poses:
            raise RuntimeError(
                f"No {self.object_type} object is available in the observation."
            )
        if not tray_poses:
            raise RuntimeError("No robot tray is available in the observation.")
        result = torch.zeros_like(
            is_object_inside_target(
                object_poses[0],
                tray_poses[0],
                thresh=self.thresh,
                keep_tensor=True,
            ),
            dtype=torch.int32,
        )
        for object_pose in object_poses:
            for tray_pose in tray_poses:
                result = torch.maximum(
                    result,
                    is_object_inside_target(
                        object_pose,
                        tray_pose,
                        thresh=self.thresh,
                        keep_tensor=True,
                    ).int(),
                )
        return result


class ObjectTypeInHallGoal(BaseGoal):
    def __init__(self, object_type: str, hall: str, thresh: float = 0.25) -> None:
        super().__init__(
            "ObjectTypeInHallGoal",
            f"object_type={object_type},hall={hall}",
        )
        self.object_type = object_type
        self.hall = hall
        self.thresh = thresh

    def verify(self, obs: Dict) -> torch.Tensor:
        extra = obs["extra"]
        object_poses = [
            pose
            for name, pose in extra.items()
            if name.startswith(self.object_type + "_")
        ]
        if not object_poses:
            raise RuntimeError(
                f"No {self.object_type} object is available in the observation."
            )
        result = torch.zeros_like(
            is_object_inside_target(
                object_poses[0],
                extra[self.hall],
                thresh=self.thresh,
                keep_tensor=True,
            ),
            dtype=torch.int32,
        )
        for object_pose in object_poses:
            result = torch.maximum(
                result,
                is_object_inside_target(
                    object_pose,
                    extra[self.hall],
                    thresh=self.thresh,
                    keep_tensor=True,
                ).int(),
            )
        return result


class PickObjectStage(BaseTaskStage):
    target_tool_calls = 3
    max_tool_calls = None

    def __init__(
        self,
        object_type: str,
        source_hall: str,
        initial_target_hall: str,
        entry_transition: BaseStageEnvironmentTransition | None = None,
    ) -> None:
        self.object_type = object_type
        self.source_hall = source_hall
        self.initial_target_hall = initial_target_hall
        super().__init__(
            goals=[ObjectTypeOnTrayGoal(object_type)],
            stage_goal_description=(
                f"Pick up one {object_type} from {source_hall} before "
                f"delivering it to {initial_target_hall}."
            ),
            stage_input=StageInput(
                instruction=UserInstruction(
                    f"Take one {object_type} from {source_hall} and deliver "
                    f"it to {initial_target_hall}."
                ),
                flag_answer_to_user=False,
            ),
            entry_transition=entry_transition,
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "object_type": self.object_type,
            "source_hall": self.source_hall,
            "initial_target_hall": self.initial_target_hall,
            "entry_transition": self.entry_transition,
        }


class RedirectPickedObjectStage(BaseTaskStage):
    target_tool_calls = 3
    max_tool_calls = None

    def __init__(self, object_type: str, redirected_hall: str) -> None:
        self.object_type = object_type
        self.redirected_hall = redirected_hall
        super().__init__(
            goals=[ObjectTypeInHallGoal(object_type, redirected_hall)],
            stage_goal_description=(
                f"Deliver the selected {object_type} to the updated destination "
                f"{redirected_hall}."
            ),
            stage_input=StageInput(
                instruction=UserInstruction(
                    f"Change of plan: deliver that {object_type} to "
                    f"{redirected_hall} instead."
                ),
                flag_answer_to_user=True,
                linked_to_prev=True,
            ),
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "object_type": self.object_type,
            "redirected_hall": self.redirected_hall,
        }


@dataclass(frozen=True)
class RedirectParameters:
    next_delivery_layout: Tuple[Tuple[str, Tuple[str, ...]], ...]
    object_name: str
    object_type: str
    source_hall: str
    initial_target_hall: str
    redirected_hall: str
    initial_instruction: str
    redirect_instruction: str


class RedirectObjectInterruptionRequest(BaseRequest[RedirectParameters]):
    def __init__(self, sampling_weight: float = INTERRUPTION_WEIGHT) -> None:
        super().__init__()
        if sampling_weight < 0:
            raise ValueError("sampling_weight must be non-negative.")
        self.weight = sampling_weight

    def sampling_weight(self, state: TaskState) -> float:
        return self.weight if get_available_delivery_types(state) else 0

    def sample_parameters(self, state: TaskState) -> RedirectParameters:
        reception_halls = state.attributes.get("reception_halls", [])
        storage_halls = state.attributes.get("storage_halls", [])
        if len(reception_halls) != 2 or len(storage_halls) != 2:
            raise RuntimeError(
                "Object redirection requires two reception and two storage halls."
            )
        layout = sample_next_delivery_layout(
            state.properties.get(DELIVERY_LAYOUT_KEY, {}),
            reception_halls,
        )
        available_objects = [
            name for hall in reception_halls for name in layout.get(hall, [])
        ]
        selected_type = get_object_type(random.choice(available_objects))
        selected_objects = [
            name for name in available_objects if get_object_type(name) == selected_type
        ]
        source_candidates = [
            hall
            for hall in HALLS
            if sum(get_object_type(name) != selected_type for name in layout[hall])
            + len(selected_objects)
            <= 4
        ]
        reception_candidates = [
            hall for hall in source_candidates if hall in reception_halls
        ]
        source_hall = random.choice(reception_candidates or source_candidates)
        for hall in HALLS:
            layout[hall] = [name for name in layout[hall] if name not in selected_objects]
        layout[source_hall].extend(selected_objects)
        target_candidates = [
            hall for hall in HALLS if hall != source_hall and len(layout[hall]) < 4
        ]
        if len(target_candidates) < 2:
            raise RuntimeError("Two destination halls need a free place for redirection.")
        object_name = random.choice(selected_objects)
        initial_target, redirected_hall = random.sample(target_candidates, k=2)
        initial_instruction = (
            f"Take one {selected_type} from {source_hall} and deliver it to {initial_target}."
        )
        redirect_instruction = (
            f"Change of plan: deliver that {selected_type} to {redirected_hall} instead."
        )
        return RedirectParameters(
            freeze_layout(layout),
            object_name,
            selected_type,
            source_hall,
            initial_target,
            redirected_hall,
            initial_instruction,
            redirect_instruction,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: RedirectParameters,
    ) -> List[BaseTaskStage]:
        stages = [
            PickObjectStage(
                parameters.object_type,
                parameters.source_hall,
                parameters.initial_target_hall,
            ),
            RedirectPickedObjectStage(
                parameters.object_type,
                parameters.redirected_hall,
            ),
        ]
        stages[0].entry_transition = ResetHallDeliveryTransition(
            delivery_layout=thaw_layout(parameters.next_delivery_layout)
        )
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: RedirectParameters,
    ) -> TaskState:
        state.properties[DELIVERY_LAYOUT_KEY] = thaw_layout(
            parameters.next_delivery_layout
        )
        return state


@dataclass(frozen=True)
class AdditionalDeliveryParameters:
    next_delivery_layout: Tuple[Tuple[str, Tuple[str, ...]], ...]
    assignment: Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...]
    object_targets: Tuple[Tuple[str, str], ...]
    first_hall: str
    second_hall: str
    completed_before_interruption: int
    first_instruction: str
    second_instruction: str


class AdditionalDeliveryInterruptionRequest(
    BaseRequest[AdditionalDeliveryParameters]
):
    def __init__(self, sampling_weight: float = INTERRUPTION_WEIGHT) -> None:
        super().__init__()
        if sampling_weight < 0:
            raise ValueError("sampling_weight must be non-negative.")
        self.weight = sampling_weight

    def sampling_weight(self, state: TaskState) -> float:
        return self.weight if get_available_delivery_types(state) else 0

    def sample_parameters(self, state: TaskState) -> AdditionalDeliveryParameters:
        reception_halls = state.attributes.get("reception_halls", [])
        storage_halls = state.attributes.get("storage_halls", [])
        known_types = state.attributes.get("object_classes", [])
        if len(reception_halls) != 2 or not storage_halls:
            raise RuntimeError("Additional delivery requires two reception halls and storage.")
        layout = sample_next_delivery_layout(
            state.properties.get(DELIVERY_LAYOUT_KEY, {}),
            reception_halls,
        )
        first_hall, second_hall = reception_halls
        first_objects = layout[first_hall]
        second_objects = layout[second_hall]
        if not first_objects or not second_objects:
            raise RuntimeError("Both deliveries must contain at least one object.")
        type_hall = state.relations.get(TYPE_HALL_RELATION, {})
        delivered_types = {
            get_object_type(name) for name in [*first_objects, *second_objects]
        }
        if all(type_hall.get(object_type) in storage_halls for object_type in delivered_types):
            effective_assignment = {
                object_type: type_hall[object_type] for object_type in delivered_types
            }
            first_instruction = (
                f"A delivery has arrived in {first_hall}. Sort its objects "
                "into their assigned storage halls."
            )
            second_instruction = (
                f"There is another delivery in {second_hall} too. Sort "
                "those objects using the same assignments."
            )
        else:
            target_hall = random.choice(storage_halls)
            effective_assignment = {
                object_type: target_hall for object_type in delivered_types
            }
            first_instruction = (
                f"A delivery has arrived in {first_hall}. Send all its objects to {target_hall}."
            )
            second_instruction = (
                f"There is another delivery in {second_hall} too. Send all of those objects "
                f"to {target_hall} as well."
            )
        assignment = {
            hall: {object_type: 0 for object_type in known_types}
            for hall in storage_halls
        }
        targets = {}
        for name in [*first_objects, *second_objects]:
            target = effective_assignment[get_object_type(name)]
            assignment[target][get_object_type(name)] += 1
            targets[name] = target
        return AdditionalDeliveryParameters(
            freeze_layout(layout),
            freeze_typed_assignment(assignment),
            tuple(targets.items()),
            first_hall,
            second_hall,
            random.randint(1, len(first_objects)),
            first_instruction,
            second_instruction,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: AdditionalDeliveryParameters,
    ) -> List[BaseTaskStage]:
        stages = build_delivery_stages(
            thaw_typed_assignment(parameters.assignment),
            parameters.first_instruction,
        )
        interruption_stage = stages[parameters.completed_before_interruption]
        interruption_stage.stage_input.instruction = UserInstruction(
            parameters.second_instruction
        )
        interruption_stage.stage_input.linked_to_prev = True
        stages[0].entry_transition = ResetHallDeliveryTransition(
            delivery_layout=thaw_layout(parameters.next_delivery_layout)
        )
        for stage in stages:
            stage.global_parameters.reset_at_end = False
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: AdditionalDeliveryParameters,
    ) -> TaskState:
        state.properties[DELIVERY_LAYOUT_KEY] = thaw_layout(
            parameters.next_delivery_layout
        )
        return state

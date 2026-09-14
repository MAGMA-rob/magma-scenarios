from typing import Dict, List, Optional, Sequence

import torch

from magma_core.simulation.data_structures import (
    EmptyInstruction,
    Log,
    StageInput,
    UserInstruction,
)
from magma_core.simulation.goals import BaseGoal
from magma_core.simulation.stage import (
    BaseStageEnvironmentTransition,
    BaseTaskStage,
    StageGlobalParameters,
)
from magma_core.simulation.utils.env_utils import is_object_inside_target

from .attributes import FOOD_OBJECTS
from .recipe_engine import MAX_RECIPE_SIZE, MIN_RECIPE_SIZE


class ExactTrayCompositionGoal(BaseGoal):
    def __init__(
        self,
        expected_objects: Sequence[str],
        threshold: float = 0.2,
    ) -> None:
        expected_objects = list(expected_objects)
        if len(expected_objects) < MIN_RECIPE_SIZE:
            raise ValueError(
                f"A tray composition requires at least {MIN_RECIPE_SIZE} objects"
            )
        if len(expected_objects) > MAX_RECIPE_SIZE:
            raise ValueError(
                f"A tray composition cannot exceed {MAX_RECIPE_SIZE} objects"
            )
        if len(set(expected_objects)) != len(expected_objects):
            raise ValueError("A tray composition cannot contain duplicate objects")
        unknown_objects = set(expected_objects) - set(FOOD_OBJECTS)
        if unknown_objects:
            raise ValueError(f"Unknown food objects: {sorted(unknown_objects)}")

        super().__init__("ExactTrayComposition", ",".join(expected_objects))
        self.expected_objects = expected_objects
        self.threshold = threshold

    def verify(self, obs: Dict) -> torch.Tensor:
        extra = obs["extra"]
        tray_pose = extra["tray"]
        nb_envs = 1 if tray_pose.ndim == 1 else tray_pose.shape[0]
        matches = torch.ones(
            nb_envs,
            dtype=torch.bool,
            device=tray_pose.device,
        )
        expected = set(self.expected_objects)

        for food in FOOD_OBJECTS:
            inside = is_object_inside_target(
                extra[food],
                tray_pose,
                thresh=self.threshold,
                keep_tensor=True,
            )
            matches &= inside if food in expected else ~inside

        return matches.int()


class TrayCompositionProgressGoal(BaseGoal):
    def __init__(
        self,
        expected_objects: Sequence[str],
        minimum: int,
        threshold: float = 0.2,
    ) -> None:
        expected_objects = list(expected_objects)
        if not MIN_RECIPE_SIZE <= len(expected_objects) <= MAX_RECIPE_SIZE:
            raise ValueError(
                "A progress recipe must contain between "
                f"{MIN_RECIPE_SIZE} and {MAX_RECIPE_SIZE} objects"
            )
        if len(set(expected_objects)) != len(expected_objects):
            raise ValueError("A progress recipe cannot contain duplicate objects")
        if not 1 <= minimum < len(expected_objects):
            raise ValueError(
                "A progress goal minimum must be between 1 and recipe size - 1"
            )
        unknown_objects = set(expected_objects) - set(FOOD_OBJECTS)
        if unknown_objects:
            raise ValueError(f"Unknown food objects: {sorted(unknown_objects)}")

        super().__init__("TrayCompositionProgress", f"minimum={minimum}")
        self.expected_objects = expected_objects
        self.minimum = minimum
        self.threshold = threshold

    def verify(self, obs: Dict) -> torch.Tensor:
        extra = obs["extra"]
        tray_pose = extra["tray"]
        nb_envs = 1 if tray_pose.ndim == 1 else tray_pose.shape[0]
        count = torch.zeros(
            nb_envs,
            dtype=torch.int32,
            device=tray_pose.device,
        )
        for food in self.expected_objects:
            count += is_object_inside_target(
                extra[food],
                tray_pose,
                thresh=self.threshold,
                keep_tensor=True,
            ).int()
        return count.ge(self.minimum).int()


class PackagingCompositionProgressStage(BaseTaskStage):
    target_tool_calls = 2
    max_tool_calls = 6

    def __init__(
        self,
        expected_objects: Sequence[str],
        minimum: int,
        instruction: str = "none",
        threshold: float = 0.2,
        entry_transition: Optional[BaseStageEnvironmentTransition] = None,
    ) -> None:
        self.expected_objects = list(expected_objects)
        self.minimum = minimum
        self.instruction = instruction
        self.threshold = threshold
        stage_instruction = (
            EmptyInstruction()
            if instruction == "none"
            else UserInstruction(instruction)
        )

        super().__init__(
            goals=[
                TrayCompositionProgressGoal(
                    self.expected_objects,
                    minimum,
                    threshold,
                )
            ],
            stage_goal_description=(
                f"At least {minimum} requested food object(s) must be on the "
                "tray, in any order."
            ),
            stage_input=StageInput(
                instruction=stage_instruction,
                flag_answer_to_user=False,
            ),
            global_parameters=StageGlobalParameters(reset_at_end=False),
            entry_transition=entry_transition,
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "expected_objects": self.expected_objects.copy(),
            "minimum": self.minimum,
            "instruction": self.instruction,
            "threshold": self.threshold,
            "entry_transition": self.entry_transition,
        }


class PackagingCompositionStage(BaseTaskStage):
    def __init__(
        self,
        expected_objects: Sequence[str],
        instruction: str,
        linked_to_prev: bool = False,
        threshold: float = 0.2,
        entry_transition: Optional[BaseStageEnvironmentTransition] = None,
        placed_object_count: int = 0,
        replacement_may_be_placed: bool = False,
    ) -> None:
        self.expected_objects = list(expected_objects)
        self.instruction = instruction
        self.threshold = threshold
        self.placed_object_count = placed_object_count
        self.replacement_may_be_placed = replacement_may_be_placed
        if not 0 <= placed_object_count < len(self.expected_objects):
            raise ValueError(
                "placed_object_count must be between 0 and recipe size - 1"
            )
        self.target_tool_calls = 2 * (
            len(self.expected_objects) - placed_object_count
        ) + 1
        self.max_tool_calls = self.target_tool_calls *2
        stage_instruction = (
            EmptyInstruction()
            if instruction == "none"
            else UserInstruction(instruction)
        )

        super().__init__(
            goals=[ExactTrayCompositionGoal(self.expected_objects, threshold)],
            stage_goal_description=(
                "The tray must contain exactly these food objects: "
                f"{', '.join(self.expected_objects)}. The valid_plate tool must "
                "then report exactly this composition."
            ),
            stage_input=StageInput(
                instruction=stage_instruction,
                flag_answer_to_user=True,
                linked_to_prev=linked_to_prev,
            ),
            global_parameters=StageGlobalParameters(reset_at_end=False),
            entry_transition=entry_transition,
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "expected_objects": self.expected_objects.copy(),
            "instruction": self.instruction,
            "linked_to_prev": self.stage_input.linked_to_prev,
            "threshold": self.threshold,
            "entry_transition": self.entry_transition,
            "placed_object_count": self.placed_object_count,
            "replacement_may_be_placed": self.replacement_may_be_placed,
        }

    def verif_log_completion(
        self,
        stage_log: List[Log],
        full_log: List[Log],
    ) -> int:
        validation_logs = [
            log for log in stage_log if log.function == "valid_plate"
        ]
        if not validation_logs:
            return 0
        if len(validation_logs) != 1:
            return -1

        content = validation_logs[0].content
        if not isinstance(content, dict):
            return -1
        tray_objects = []
        for objects in content.values():
            if not isinstance(objects, list):
                return -1
            tray_objects.extend(objects)
        return 1 if sorted(tray_objects) == sorted(self.expected_objects) else -1

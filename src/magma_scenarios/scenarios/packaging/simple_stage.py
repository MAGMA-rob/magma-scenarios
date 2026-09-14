from typing import Dict

import torch

from magma_core.simulation.data_structures import StageInput, UserInstruction
from magma_core.simulation.goals import BaseGoal
from magma_core.simulation.stage import BaseTaskStage, StageGlobalParameters
from magma_core.simulation.utils.env_utils import is_object_inside_target

from .attributes import drinks, fruits, main_course


class AtLeastOneObjectOnTrayGoal(BaseGoal):
    def __init__(self, threshold: float = 0.2) -> None:
        super().__init__("AtLeastOneObjectOnTray", "minimum=1")
        self.threshold = threshold

    def verify(self, obs: Dict) -> torch.Tensor:
        extra = obs["extra"]
        tray_pose = extra["tray"]
        nb_envs = 1 if tray_pose.ndim == 1 else tray_pose.shape[0]
        result = torch.zeros(
            nb_envs,
            dtype=torch.int32,
            device=tray_pose.device,
        )

        for name in [*main_course, *fruits, *drinks]:
            if name not in extra:
                continue
            result = torch.maximum(
                result,
                is_object_inside_target(
                    extra[name],
                    tray_pose,
                    thresh=self.threshold,
                    keep_tensor=True,
                ).int(),
            )

        return result


class SimplePutOnTrayStage(BaseTaskStage):
    """Place at least one food item on the tray."""

    target_tool_calls = 3
    max_tool_calls = 4

    def __init__(self, threshold: float = 0.2) -> None:
        super().__init__(
            goals=[AtLeastOneObjectOnTrayGoal(threshold)],
            stage_goal_description="Place at least one food item on the tray.",
            stage_input=StageInput(
                instruction=UserInstruction(
                    "Place at least one food item from the table on the tray."
                ),
                flag_answer_to_user=True,
            ),
            global_parameters=StageGlobalParameters(reset_at_end=True),
        )
        self.threshold = threshold

    def _to_spec_arguments(self) -> Dict:
        return {"threshold": self.threshold}

from copy import deepcopy
from dataclasses import dataclass
import random
from typing import List, Tuple

from magma_core.simulation.data_structures import (
    EmptyInstruction,
    StageInput,
    UserInstruction,
)
from magma_core.simulation.stage import BaseTaskStage, ModifAttributesBaseStage
from magma_core.simulation.state import TaskState
from magma_scenarios.templates.requests.interact_request import (
    AttributesModificationParameters,
)
from magma_scenarios.templates.requests import (
    AddValueToListRequest,
    RemoveValueToListRequest,
)


@dataclass(frozen=True)
class AreaModificationParameters(AttributesModificationParameters):
    areas: Tuple[str, ...]
    instruction: str


class AddAreas(AddValueToListRequest):
    def __init__(self, all_areas: List[str], max_update: int = 2) -> None:
        super().__init__(
            modifiable_task_attributes={"target_areas": all_areas},
            max_update=max_update,
        )

    def sample_parameters(self, state: TaskState) -> AreaModificationParameters:
        update_count = random.randint(1, self.max_update)
        attributes = deepcopy(state.attributes)
        modified_areas = []
        for _ in range(update_count):
            sampled = self._get_random_key_value(["target_areas"], attributes)
            if sampled is None:
                break
            attributes[sampled[0]].append(sampled[1])
            modified_areas.append(sampled[1])
        if not modified_areas:
            raise RuntimeError("No target area is available to add.")
        instruction = f"Please add {' and '.join(modified_areas)} to your known areas"
        return AreaModificationParameters(
            attributes,
            tuple(modified_areas),
            instruction,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: AreaModificationParameters,
    ) -> list[BaseTaskStage]:
        stages = []
        instruction = UserInstruction(parameters.instruction)
        for index, area in enumerate(parameters.areas):
            stages.append(
                ModifAttributesBaseStage(
                    mode="ADD",
                    stage_input=StageInput(
                        instruction,
                        index == len(parameters.areas) - 1,
                    ),
                    val_name=area,
                    att_name="target_areas",
                )
            )
            instruction = EmptyInstruction()
        stages[-1].target_tool_calls += 1
        stages[-1].max_tool_calls = stages[-1].target_tool_calls
        return stages


class RemoveAreas(RemoveValueToListRequest):
    def __init__(self, max_update: int = 2) -> None:
        super().__init__(["target_areas"], max_update=max_update)

    def sampling_weight(self, state: TaskState) -> float:
        if len(state.attributes.get("target_areas", [])) <= 1:
            return 0
        return super().sampling_weight(state)

    def sample_parameters(self, state: TaskState) -> AreaModificationParameters:
        update_count = random.randint(1, self.max_update)
        attributes = deepcopy(state.attributes)
        modified_areas = []
        for _ in range(update_count):
            if len(attributes["target_areas"]) == 1:
                break
            sampled = self._get_random_key_value(["target_areas"], attributes)
            if sampled is None:
                break
            attributes[sampled[0]].remove(sampled[1])
            modified_areas.append(sampled[1])
        if not modified_areas:
            raise RuntimeError("No target area is available to remove.")
        instruction = (
            f"Please remove {' and '.join(modified_areas)} from your knowledge base."
        )
        return AreaModificationParameters(
            attributes,
            tuple(modified_areas),
            instruction,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: AreaModificationParameters,
    ) -> list[BaseTaskStage]:
        stages = []
        instruction = UserInstruction(parameters.instruction)
        for index, area in enumerate(parameters.areas):
            stages.append(
                ModifAttributesBaseStage(
                    mode="REMOVE",
                    stage_input=StageInput(
                        instruction,
                        index == len(parameters.areas) - 1,
                    ),
                    val_name=area,
                    att_name="target_areas",
                )
            )
            instruction = EmptyInstruction()
        stages[-1].target_tool_calls += 1
        stages[-1].max_tool_calls = stages[-1].target_tool_calls
        return stages

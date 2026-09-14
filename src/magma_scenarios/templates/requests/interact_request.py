import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from magma_core.simulation.constraints import AttributesModifConstraint, BaseConstraint
from magma_core.simulation.stage import BaseTaskStage, ConstraintBaseStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests.base_request import BaseRequest


@dataclass(frozen=True)
class ConstraintParameters:
    constraints: Tuple[BaseConstraint, ...]
    instruction: str

    def __init__(
        self,
        constraints: Sequence[BaseConstraint],
        instruction: str,
    ) -> None:
        if not constraints:
            raise ValueError("Constraint parameters must define constraints.")
        if not instruction:
            raise ValueError("Constraint parameters must define an instruction.")
        object.__setattr__(self, "constraints", tuple(constraints))
        object.__setattr__(self, "instruction", instruction)


class BaseConstraintRequest(BaseRequest[ConstraintParameters]):
    """Common runtime behavior for permanent rules."""

    reset_at_end: bool = True

    def create_stages(
        self,
        state: TaskState,
        parameters: ConstraintParameters,
    ) -> list[BaseTaskStage]:
        return [
            ConstraintBaseStage(
                parameters.instruction,
                reset_at_end=self.reset_at_end,
            )
        ]

    def apply_request(
        self,
        state: TaskState,
        parameters: ConstraintParameters,
    ) -> TaskState:
        for constraint in parameters.constraints:
            constraint.apply(state)
        return state


@dataclass(frozen=True)
class AttributesModificationParameters:
    attributes: Mapping[str, Any]

    def __init__(self, attributes: Mapping[str, Any]) -> None:
        object.__setattr__(self, "attributes", copy.deepcopy(dict(attributes)))


class BaseAttributesModifRequest(
    BaseRequest[AttributesModificationParameters]
):
    """Apply a sampled attribute snapshot after its runtime stages."""

    def __init__(self, modifiable_task_attributes: Mapping[str, Any]) -> None:
        super().__init__()
        self.attributes = modifiable_task_attributes

    def sampling_weight(self, state: TaskState) -> float:
        for entity_name, entity_value in self.attributes.items():
            if type(entity_value) != type(state.attributes[entity_name]):
                raise ValueError(
                    "Incompatible task and state attribute types for "
                    f"{entity_name!r}."
                )
        return 1

    def apply_request(
        self,
        state: TaskState,
        parameters: AttributesModificationParameters,
    ) -> TaskState:
        AttributesModifConstraint(copy.deepcopy(parameters.attributes)).apply(state)
        return state

    def force_state_recompute(self) -> bool:
        return True

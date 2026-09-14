from dataclasses import dataclass
import random
from typing import Dict, List, Tuple

from magma_core.simulation.data_structures import UserInstruction
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.utils.text_utils import join_with_and

from ..warehouse_stages import build_object_to_zone_stages


@dataclass(frozen=True)
class WarehouseInterruptionParameters:
    all_areas: Tuple[str, ...]
    initial_assignment: Tuple[Tuple[str, str], ...]
    priority_assignment: Tuple[Tuple[str, str], ...]
    initial_instruction: str
    priority_instruction: str
    initial_has_constraint: bool
    priority_has_constraint: bool
    interrupt_after: int

    @property
    def selected_objects(self) -> Tuple[str, ...]:
        return tuple(
            object_name
            for object_name, _ in (
                self.initial_assignment + self.priority_assignment
            )
        )


class WarehouseSortingInterruptionRequest(
    BaseRequest[WarehouseInterruptionParameters]
):
    """Interrupt a sorting cycle after one of its objects is completed."""

    def sampling_weight(self, state: TaskState) -> float:
        return float(
            len(state.attributes.get("objects", [])) >= 3
            and len(state.attributes.get("target_areas", [])) > 0
        )

    def _create_assignment(
        self,
        objects: List[str],
        areas: List[str],
        default_assignments: Dict[str, str],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        assignment = {}
        explicit_assignment = {}
        for object_name in objects:
            default_area = default_assignments.get(object_name)
            if default_area in areas and random.random() < 0.5:
                assignment[object_name] = default_area
            else:
                area = random.choice(areas)
                assignment[object_name] = area
                explicit_assignment[object_name] = area
        return assignment, explicit_assignment

    def _cycle_description(
        self,
        objects: List[str],
        explicit_assignment: Dict[str, str],
    ) -> str:
        return join_with_and(
            [
                (
                    f"{object_name} to {explicit_assignment[object_name]}"
                    if object_name in explicit_assignment
                    else object_name
                )
                for object_name in objects
            ]
        )

    def sample_parameters(
        self,
        state: TaskState,
    ) -> WarehouseInterruptionParameters:
        all_objects = state.attributes.get("objects", []).copy()
        all_areas = state.attributes.get("target_areas", []).copy()
        if len(all_objects) < 3:
            raise RuntimeError(
                "WarehouseSortingInterruptionRequest requires at least three objects."
            )
        if not all_areas:
            raise RuntimeError(
                "WarehouseSortingInterruptionRequest requires at least one target area."
            )
        random.shuffle(all_objects)
        selected_count = random.randint(3, len(all_objects))
        selected_objects = all_objects[:selected_count]
        initial_count = random.randint(2, selected_count - 1)
        initial_objects = selected_objects[:initial_count]
        priority_objects = selected_objects[initial_count:]
        defaults = state.relations.get("object_area", {})
        initial_assignment, initial_explicit = self._create_assignment(
            initial_objects, all_areas, defaults
        )
        priority_assignment, priority_explicit = self._create_assignment(
            priority_objects, all_areas, defaults
        )
        initial_instruction = (
            "Launch a sorting cycle for "
            f"{self._cycle_description(initial_objects, initial_explicit)}."
        )
        priority_instruction = (
            "Urgent interruption: stop the current cycle and launch a priority "
            "cycle for "
            f"{self._cycle_description(priority_objects, priority_explicit)}. "
            "Complete it first, then resume the interrupted cycle."
        )
        return WarehouseInterruptionParameters(
            tuple(all_areas),
            tuple(initial_assignment.items()),
            tuple(priority_assignment.items()),
            initial_instruction,
            priority_instruction,
            bool(initial_explicit),
            bool(priority_explicit),
            random.randint(1, len(initial_objects) - 1),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: WarehouseInterruptionParameters,
    ) -> list[BaseTaskStage]:
        initial_assignment = dict(parameters.initial_assignment)
        priority_assignment = dict(parameters.priority_assignment)
        initial_stages = build_object_to_zone_stages(
            assignment=initial_assignment,
            known_areas=list(parameters.all_areas),
            instruction=UserInstruction(
                parameters.initial_instruction,
                has_constraint=parameters.initial_has_constraint,
            ),
            flag_answer=True,
            reset_at_end=True,
            final_message_decision=True,
        )
        priority_stages = build_object_to_zone_stages(
            assignment=priority_assignment,
            known_areas=list(parameters.all_areas),
            instruction=UserInstruction(
                parameters.priority_instruction,
                has_constraint=parameters.priority_has_constraint,
            ),
            flag_answer=False,
            linked_to_prev=True,
            reset_at_end=False,
        )
        return (
            initial_stages[: parameters.interrupt_after]
            + priority_stages
            + initial_stages[parameters.interrupt_after :]
        )

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import random

from magma_core.simulation.data_structures import EmptyInstruction, UserInstruction
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.utils.text_utils import is_or_are, join_with_and
from magma_scenarios.templates.stages import ForbiddenElemStage

from ..warehouse_stages import build_object_to_zone_stages


@dataclass(frozen=True)
class TemporaryCycle:
    objects: Tuple[str, ...]
    target_area: str


@dataclass(frozen=True)
class TemporaryCycleParameters:
    cycles: Tuple[TemporaryCycle, ...]
    all_areas: Tuple[str, ...]
    instruction: str
    refusal_message: Optional[str] = None
    refusal_stage_answer: Optional[str] = None


class TemporaryObjectAssignmentCycleRequest(
    BaseRequest[TemporaryCycleParameters]
):
    """Launch one or more one-shot assignment cycles without updating defaults."""

    def __init__(
            self,
            max_object_per_cycle_request: int = 3,
            max_nb_cycle : int = 2,
            all_objects_probability: float = 0.7,
        ) -> None:
        super().__init__()
        self.max_object = max_object_per_cycle_request
        self.max_nb_cycle = max_nb_cycle
        self.all_objects_probability = all_objects_probability

    def sampling_weight(self, state: TaskState) -> float:
        if len(state.attributes.get("objects", [])) <= 0:
            return 0
        if len(state.attributes.get("target_areas", [])) <= 0:
            return 0
        if state.properties.get("object_area_needs_application", False):
            return 0.5
        return 2

    def _format_cycle_assignment(
            self,
            objects_to_sort: List[str],
            target_area: str,
            all_objects: List[str],
        ) -> str:
        if len(objects_to_sort) == len(all_objects):
            return f"all objects to {target_area}"
        return f"{join_with_and(objects_to_sort)} to {target_area}"

    def _build_instruction(
            self,
            cycle_plan: List[Dict[str, Any]],
            all_objects: List[str],
        ) -> UserInstruction:
        first_assignment = self._format_cycle_assignment(
            cycle_plan[0]["objects"],
            cycle_plan[0]["target_area"],
            all_objects,
        )

        if len(cycle_plan) == 1:
            objects_to_sort = cycle_plan[0]["objects"]
            target_area = cycle_plan[0]["target_area"]
            if len(objects_to_sort) == len(all_objects):
                templates = (
                    f"Launch a cycle sending all objects to {target_area}.",
                    f"For this cycle, send every object to {target_area}.",
                )
            else:
                obj_text = join_with_and(objects_to_sort)
                templates = (
                    f"Launch a cycle sending {obj_text} to {target_area}.",
                    f"For this cycle, send {obj_text} to {target_area}.",
                )
            return UserInstruction(random.choice(templates), has_constraint=True)

        if len(cycle_plan) == 2:
            second_assignment = self._format_cycle_assignment(
                cycle_plan[1]["objects"],
                cycle_plan[1]["target_area"],
                all_objects,
            )
            templates = (
                (
                    f"I want you to launch two cycles. First, send {first_assignment}. "
                    f"Then when it's done, launch a cycle sending {second_assignment}."
                ),
                (
                    f"Please launch two cycles in sequence: first send {first_assignment}, "
                    f"then send {second_assignment}."
                ),
                (
                    f"I need a first cycle with {first_assignment}. "
                    f"Right after, launch another cycle with {second_assignment}."
                ),
            )
            return UserInstruction(random.choice(templates), has_constraint=True)

        steps = [f"First, send {first_assignment}"]
        for cycle in cycle_plan[1:-1]:
            steps.append(
                "then send "
                + self._format_cycle_assignment(
                    cycle["objects"],
                    cycle["target_area"],
                    all_objects,
                )
            )
        steps.append(
            "finally send "
            + self._format_cycle_assignment(
                cycle_plan[-1]["objects"],
                cycle_plan[-1]["target_area"],
                all_objects,
            )
        )
        return UserInstruction(
            f"I want you to launch {len(cycle_plan)} cycles. " + ", ".join(steps) + ".",
            has_constraint=True,
        )

    def _sample_cycle_plan(
            self,
            all_objects: List[str],
            all_areas: List[str],
        ) -> List[Dict[str, Any]]:
        cycle_count = random.randint(
            1,
            min(self.max_nb_cycle, len(all_objects)),
        )
        maximum_selected = min(
            len(all_objects),
            self.max_object * cycle_count,
        )
        selected_count = (
            maximum_selected
            if random.random() < self.all_objects_probability
            else random.randint(cycle_count, maximum_selected)
        )
        selected_objects = random.sample(all_objects, k=selected_count)
        cycle_objects = [[] for _ in range(cycle_count)]
        for index, object_name in enumerate(selected_objects):
            cycle_objects[index % cycle_count].append(object_name)
        return [
            {
                "objects": objects,
                "target_area": random.choice(all_areas),
            }
            for objects in cycle_objects
        ]

    def sample_parameters(self, state: TaskState) -> TemporaryCycleParameters:
        all_objects = state.attributes.get("objects", []).copy()
        all_areas = state.attributes.get("target_areas", [])
        if len(all_objects) <= 0 or len(all_areas) <= 0:
            raise RuntimeError(
                f"Failed to build the stage from {self.__class__.__name__} "
                "due to empty objects or areas"
            )

        cycle_plan = self._sample_cycle_plan(all_objects, all_areas)
        instruction = self._build_instruction(cycle_plan, all_objects)

        objects_to_sort = [
            obj
            for cycle in cycle_plan
            for obj in cycle["objects"]
        ]
        target_areas = [
            cycle["target_area"]
            for cycle in cycle_plan
        ]

        forbidden_objects = list(dict.fromkeys(
            obj
            for obj in objects_to_sort
            if obj in state.properties.get("forbidden_objects", [])
        ))
        if forbidden_objects:
            refusal_stage_answer = (
                    f"The model must inform that {join_with_and(forbidden_objects)} "
                    f"{is_or_are(forbidden_objects)} forbidden"
            )
            refusal_message = (
                        f"{join_with_and(forbidden_objects)} "
                        f"{is_or_are(forbidden_objects)} forbidden."
            )
            return TemporaryCycleParameters(
                tuple(TemporaryCycle(tuple(cycle["objects"]), cycle["target_area"]) for cycle in cycle_plan),
                tuple(all_areas),
                instruction.get_content(),
                refusal_message,
                refusal_stage_answer,
            )

        forbidden_areas = list(dict.fromkeys(
            area
            for area in target_areas
            if area in state.properties.get("forbidden_areas", [])
        ))
        if forbidden_areas:
            refusal_stage_answer = (
                    "The model must inform that "
                    f"{join_with_and(forbidden_areas)} can not be used."
            )
            return TemporaryCycleParameters(
                tuple(TemporaryCycle(tuple(cycle["objects"]), cycle["target_area"]) for cycle in cycle_plan),
                tuple(all_areas),
                instruction.get_content(),
                f"{join_with_and(forbidden_areas)} cannot be used.",
                refusal_stage_answer,
            )

        return TemporaryCycleParameters(
            tuple(TemporaryCycle(tuple(cycle["objects"]), cycle["target_area"]) for cycle in cycle_plan),
            tuple(all_areas),
            instruction.get_content(),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: TemporaryCycleParameters,
    ) -> List[BaseTaskStage]:
        instruction = UserInstruction(parameters.instruction, has_constraint=True)
        if parameters.refusal_message is not None:
            return [ForbiddenElemStage(
                instruction=instruction,
                answer=parameters.refusal_stage_answer,
            )]

        stages = []
        for i, cycle in enumerate(parameters.cycles):
            stages.extend(build_object_to_zone_stages(
                assignment={obj: cycle.target_area for obj in cycle.objects},
                known_areas=list(parameters.all_areas),
                flag_answer=i == len(parameters.cycles) - 1,
                instruction=instruction if i == 0 else EmptyInstruction(),
                linked_to_prev=i > 0,
                reset_at_end=i == len(parameters.cycles) - 1,
                final_message_decision=i == len(parameters.cycles) - 1,
            ))
        return stages

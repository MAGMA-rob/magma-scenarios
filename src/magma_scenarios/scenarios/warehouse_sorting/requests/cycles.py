from dataclasses import dataclass
import random
from typing import Dict, List, Sequence, Tuple

from magma_core.simulation.data_structures import UserInstruction
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.utils.text_utils import is_or_are
from magma_scenarios.templates.constraints import RelationAssignmentConstraint
from magma_scenarios.templates.stages import ForbiddenElemStage, MissingInformationStage

from ..warehouse_stages import build_object_to_zone_stages


OBJECT_AREA_PENDING_APPLICATION_WEIGHT = 7.0
OBJECT_AREA_PENDING_APPLICATION_MIN_WEIGHT = 0.0
OBJECT_AREA_PENDING_APPLICATION_DECAY = 1.0
OBJECT_AREA_PENDING_RULE_UPDATE_WEIGHT = 0.5
OBJECT_AREA_RULE_UPDATE_WEIGHT = 1.0
CATEGORY_CYCLE_WEIGHT = 2.5
CATEGORY_CYCLE_EMPTY_TYPE_AREA_WEIGHT = 1.5
CATEGORY_CYCLE_PENDING_OBJECT_TYPE_WEIGHT = 4.0
CATEGORY_CYCLE_PENDING_TYPE_AREA_WEIGHT = 6.0


@dataclass(frozen=True)
class SortingClarification:
    missing_assignment: Tuple[Tuple[str, str], ...]
    question: str
    answer: str


@dataclass(frozen=True)
class SortingRefusal:
    message: str
    stage_answer: str


@dataclass(frozen=True)
class CycleParameters:
    objects_to_sort: Tuple[str, ...]
    all_areas: Tuple[str, ...]
    assignment: Tuple[Tuple[str, str], ...]
    base_assignment: Tuple[Tuple[str, str], ...]
    instruction: str
    clarification: SortingClarification | None
    refusal: SortingRefusal | None
    applied_pending_forbidden_object: bool
    constraints: Tuple[RelationAssignmentConstraint, ...] = ()


class CycleRequest(BaseRequest[CycleParameters]):
    def __init__(
        self,
        max_object_per_cycle_request: int = 3,
        pending_application_weight: float = OBJECT_AREA_PENDING_APPLICATION_WEIGHT,
        pending_application_min_weight: float = OBJECT_AREA_PENDING_APPLICATION_MIN_WEIGHT,
        pending_application_decay: float = OBJECT_AREA_PENDING_APPLICATION_DECAY,
        forbidden_object_sampling_probability: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_object = max_object_per_cycle_request
        if pending_application_weight < 0:
            raise ValueError("pending_application_weight must be non-negative")
        if pending_application_min_weight < 0:
            raise ValueError("pending_application_min_weight must be non-negative")
        if not 0 < pending_application_decay <= 1:
            raise ValueError("pending_application_decay must be in (0, 1]")
        if not 0 <= forbidden_object_sampling_probability <= 1:
            raise ValueError(
                "forbidden_object_sampling_probability must be in [0, 1]"
            )
        self.pending_application_weight = pending_application_weight
        self.pending_application_min_weight = pending_application_min_weight
        self.pending_application_decay = pending_application_decay
        self.forbidden_object_sampling_probability = (
            forbidden_object_sampling_probability
        )

    def _pending_application_sampling_weight(self, state: TaskState) -> float:
        applications = state.properties.get("object_area_applications", 0)
        return max(
            self.pending_application_min_weight,
            self.pending_application_weight
            * (self.pending_application_decay ** applications),
        )

    def sampling_weight(self, state: TaskState) -> float:
        if not state.attributes.get("objects", []):
            return 0
        if not state.attributes.get("target_areas", []):
            return 0
        if state.properties.get("object_area_needs_application", False):
            return self._pending_application_sampling_weight(state)
        if state.relations.get("object_area", {}):
            return 4
        return 2

    def _sample_objects_to_sort(self, state: TaskState) -> List[str]:
        all_objects = state.attributes.get("objects", []).copy()
        random.shuffle(all_objects)
        max_objects = min(self.max_object, len(all_objects))
        nb_obj = random.randint(1, max_objects)

        object_assignments = state.relations.get("object_area", {})
        assigned_objects = [obj for obj in all_objects if obj in object_assignments]
        missing_objects = [obj for obj in all_objects if obj not in object_assignments]
        forbidden_objects = [
            obj
            for obj in state.properties.get("forbidden_objects", [])
            if obj in all_objects
        ]
        if (
            state.properties.get("forbidden_object_needs_application", False)
            and forbidden_objects
            and random.random() < self.forbidden_object_sampling_probability
        ):
            selected = [random.choice(forbidden_objects)]
            for obj in all_objects:
                if len(selected) >= nb_obj:
                    break
                if obj not in selected:
                    selected.append(obj)
            return selected

        if state.properties.get("object_area_needs_application", False):
            pending_sources = [
                obj
                for obj in state.properties.get("object_area_pending_sources", [])
                if obj in object_assignments and obj in all_objects
            ]
            if pending_sources:
                random.shuffle(pending_sources)
                max_pending_objects = min(self.max_object, len(pending_sources))
                nb_pending_obj = random.randint(1, max_pending_objects)
                return pending_sources[:nb_pending_obj]
            if assigned_objects:
                max_assigned_objects = min(self.max_object, len(assigned_objects))
                nb_assigned_obj = random.randint(1, max_assigned_objects)
                random.shuffle(assigned_objects)
                return assigned_objects[:nb_assigned_obj]

        selected: List[str] = []
        if object_assignments and missing_objects and random.random() < 0.5:
            selected.append(random.choice(missing_objects))
            if len(selected) < nb_obj and assigned_objects:
                selected.append(random.choice(assigned_objects))
        for obj in all_objects:
            if len(selected) >= nb_obj:
                break
            if obj not in selected:
                selected.append(obj)
        return selected

    def _sample_cycle_parameters(
        self,
        state: TaskState,
        objects_to_sort: Sequence[str],
        all_areas: Sequence[str],
        base_assignment: Dict[str, str] | None = None,
        constraints: Sequence[RelationAssignmentConstraint] = (),
    ) -> CycleParameters:
        base_assignment = dict(base_assignment or {})
        assignment = base_assignment.copy()
        missing_assignment: Dict[str, str] = {}
        forbidden_objects = []
        objects_with_forbidden_areas = []
        for object_name in objects_to_sort:
            if object_name in state.properties.get("forbidden_objects", []):
                forbidden_objects.append(object_name)
            if object_name not in assignment:
                target_area = state.relations.get("object_area", {}).get(
                    object_name,
                    state.properties.get("relation_default_targets", {}).get(
                        "object_area", "none"
                    ),
                )
                if target_area not in all_areas:
                    target_area = random.choice(all_areas)
                    missing_assignment[object_name] = target_area
                else:
                    assignment[object_name] = target_area
            else:
                target_area = assignment[object_name]
            if target_area in state.properties.get("forbidden_areas", []):
                objects_with_forbidden_areas.append(object_name)
        assignment.update(missing_assignment)

        object_text = " and ".join(assignment)
        instruction = f"Launch a cycle for {object_text}."
        if base_assignment:
            rule_text = ", ".join(
                f"{obj} to {area}" for obj, area in base_assignment.items()
            )
            assignment_word = "assignments" if len(base_assignment) > 1 else "assignment"
            instruction += (
                f" And consider {rule_text} as new default {assignment_word}."
            )

        refusal_message = None
        refusal_stage_answer = None
        if objects_with_forbidden_areas or forbidden_objects:
            refusal_reasons = []
            if forbidden_objects:
                forbidden_text = " and ".join(forbidden_objects)
                refusal_reasons.append(
                    f"{forbidden_text} {is_or_are(forbidden_objects)} forbidden"
                )
            if objects_with_forbidden_areas:
                area_text = " and ".join(
                    assignment[obj] for obj in objects_with_forbidden_areas
                )
                refusal_reasons.append(f"{area_text} cannot be used")
            refusal_message = " and ".join(refusal_reasons) + "."
            refusal_stage_answer = f"The model must inform that {refusal_message}"

        question = None
        resolution_input = instruction
        if missing_assignment and refusal_message is None:
            missing_objects = " and ".join(missing_assignment)
            assignment_text = ", ".join(
                f"{obj} goes to {area}"
                for obj, area in missing_assignment.items()
            )
            question = f"Which target areas should be used for {missing_objects}?"
            resolution_input = f"For this cycle, {assignment_text}."

        forbidden_set = set(state.properties.get("forbidden_objects", []))
        applied_pending_forbidden_object = (
            state.properties.get("forbidden_object_needs_application", False)
            and any(obj in forbidden_set for obj in objects_to_sort)
        )
        clarification = (
            SortingClarification(
                tuple(missing_assignment.items()),
                question,
                resolution_input,
            )
            if question is not None
            else None
        )
        refusal = (
            SortingRefusal(refusal_message, refusal_stage_answer)
            if refusal_message is not None and refusal_stage_answer is not None
            else None
        )
        return CycleParameters(
            objects_to_sort=tuple(objects_to_sort),
            all_areas=tuple(all_areas),
            assignment=tuple(assignment.items()),
            base_assignment=tuple(base_assignment.items()),
            instruction=instruction,
            clarification=clarification,
            refusal=refusal,
            applied_pending_forbidden_object=applied_pending_forbidden_object,
            constraints=tuple(constraints),
        )

    def sample_parameters(self, state: TaskState) -> CycleParameters:
        all_objects = state.attributes.get("objects", []).copy()
        all_areas = state.attributes.get("target_areas", [])
        if not all_objects or not all_areas:
            raise RuntimeError(
                f"Failed to build the stage from {self.__class__.__name__} "
                "due to empty objects or areas"
            )
        return self._sample_cycle_parameters(
            state,
            self._sample_objects_to_sort(state),
            all_areas,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: CycleParameters,
    ) -> list[BaseTaskStage]:
        stages: list[BaseTaskStage] = []
        instruction = UserInstruction(
            parameters.instruction,
            has_constraint=bool(parameters.base_assignment),
        )
        if parameters.refusal is not None:
            return [
                ForbiddenElemStage(
                    instruction=instruction,
                    answer=parameters.refusal.stage_answer,
                )
            ]

        if parameters.clarification is not None:
            missing_objects = " and ".join(
                obj for obj, _ in parameters.clarification.missing_assignment
            )
            stages.append(
                MissingInformationStage(
                    instruction=instruction,
                    answer=(
                        "The robot must ask about target areas for "
                        f"{missing_objects}"
                    ),
                )
            )
            instruction = UserInstruction(parameters.clarification.answer)

        for stage in build_object_to_zone_stages(
            assignment=dict(parameters.assignment),
            known_areas=list(parameters.all_areas),
            flag_answer=True,
            instruction=instruction,
            final_message_decision=True,
        ):
            if stages:
                stage.stage_input.linked_to_prev = True
            stages.append(stage)
        return stages


    def apply_request(
        self,
        state: TaskState,
        parameters: CycleParameters,
    ) -> TaskState:
        if state.properties.get("object_area_needs_application", False):
            state.properties["object_area_needs_application"] = False
            state.properties.pop("object_area_pending_sources", None)
            state.properties["object_area_applications"] = (
                state.properties.get("object_area_applications", 0) + 1
            )
        if parameters.applied_pending_forbidden_object:
            state.properties["forbidden_object_needs_application"] = False
        return state


class CycleWithPermanentRulesRequest(CycleRequest):
    def __init__(
        self,
        max_object_per_cycle_request: int = 3,
        max_permanent_rule: int = 2,
        pending_application_weight: float = OBJECT_AREA_PENDING_APPLICATION_WEIGHT,
        pending_application_min_weight: float = OBJECT_AREA_PENDING_APPLICATION_MIN_WEIGHT,
        pending_application_decay: float = OBJECT_AREA_PENDING_APPLICATION_DECAY,
        forbidden_object_sampling_probability: float = 0.0,
        rule_update_weight: float = OBJECT_AREA_RULE_UPDATE_WEIGHT,
        pending_rule_update_weight: float = OBJECT_AREA_PENDING_RULE_UPDATE_WEIGHT,
    ) -> None:
        super().__init__(
            max_object_per_cycle_request,
            pending_application_weight,
            pending_application_min_weight,
            pending_application_decay,
            forbidden_object_sampling_probability,
        )
        self.max_rules = max_permanent_rule
        if rule_update_weight < 0 or pending_rule_update_weight < 0:
            raise ValueError("rule update weights must be non-negative")
        self.rule_update_weight = rule_update_weight
        self.pending_rule_update_weight = pending_rule_update_weight

    def sampling_weight(self, state: TaskState) -> float:
        if state.properties.get("object_area_needs_application", False):
            return self.pending_rule_update_weight
        if not state.relations.get("object_area", {}):
            return 3
        return self.rule_update_weight

    def sample_parameters(self, state: TaskState) -> CycleParameters:
        all_objects = state.attributes.get("objects", []).copy()
        all_areas = state.attributes.get("target_areas", [])
        if not all_objects or not all_areas:
            raise RuntimeError(
                f"Failed to build the stage from {self.__class__.__name__} "
                "due to empty objects or areas"
            )
        nb_obj = random.randint(1, min(self.max_object, len(all_objects)))
        random.shuffle(all_objects)
        nb_rules = random.randint(1, min(nb_obj, self.max_rules))
        assignment: Dict[str, str] = {}
        constraints = []
        for index in range(nb_rules):
            area = random.choice(all_areas)
            object_name = all_objects[index]
            assignment[object_name] = area
            constraints.append(
                RelationAssignmentConstraint(
                    source_value=object_name,
                    target_value=area,
                    relation_key="object_area",
                    source_attribute_key="objects",
                    target_attribute_key="target_areas",
                )
            )
        return self._sample_cycle_parameters(
            state,
            all_objects[:nb_obj],
            all_areas,
            assignment,
            constraints,
        )

    def apply_request(
        self,
        state: TaskState,
        parameters: CycleParameters,
    ) -> TaskState:
        for constraint in parameters.constraints:
            constraint.apply(state)
        if parameters.constraints:
            state.properties["object_area_needs_application"] = True
            state.properties["object_area_pending_sources"] = [
                constraint.source_value for constraint in parameters.constraints
            ]
        return state


@dataclass(frozen=True)
class CategoryClarification:
    missing_assignment: Tuple[Tuple[str, str], ...]
    question: str
    answer: str


@dataclass(frozen=True)
class CategoryCycleParameters:
    selected_types: Tuple[str, ...]
    all_areas: Tuple[str, ...]
    assignment: Tuple[Tuple[str, str], ...]
    initial_instruction: str
    clarification: CategoryClarification | None


class CycleByCategoriesRequest(BaseRequest[CategoryCycleParameters]):
    def __init__(
        self,
        max_categories_per_cycle_request: int = 2,
        cycle_weight: float = CATEGORY_CYCLE_WEIGHT,
        empty_type_area_weight: float = CATEGORY_CYCLE_EMPTY_TYPE_AREA_WEIGHT,
        pending_object_type_weight: float = CATEGORY_CYCLE_PENDING_OBJECT_TYPE_WEIGHT,
        pending_type_area_weight: float = CATEGORY_CYCLE_PENDING_TYPE_AREA_WEIGHT,
    ) -> None:
        super().__init__()
        self.max_categories = max_categories_per_cycle_request
        for name, value in (
            ("cycle_weight", cycle_weight),
            ("empty_type_area_weight", empty_type_area_weight),
            ("pending_object_type_weight", pending_object_type_weight),
            ("pending_type_area_weight", pending_type_area_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        self.cycle_weight = cycle_weight
        self.empty_type_area_weight = empty_type_area_weight
        self.pending_object_type_weight = pending_object_type_weight
        self.pending_type_area_weight = pending_type_area_weight

    def _get_known_types(self, state: TaskState) -> List[str]:
        known_objects = set(state.attributes.get("objects", []))
        known_types = []
        for object_name, object_type in state.relations.get("object_type", {}).items():
            if object_name in known_objects and object_type not in known_types:
                known_types.append(object_type)
        return known_types

    def sampling_weight(self, state: TaskState) -> float:
        if state.properties.get("forbidden_objects", []) or state.properties.get(
            "forbidden_areas", []
        ):
            return 0
        if not self._get_known_types(state):
            return 0
        if not state.attributes.get("objects", []) or not state.attributes.get(
            "target_areas", []
        ):
            return 0
        if state.properties.get("type_area_needs_application", False):
            return self.pending_type_area_weight
        if state.properties.get("object_type_needs_application", False):
            return self.pending_object_type_weight
        if state.relations.get("type_area", {}):
            return self.cycle_weight
        return self.empty_type_area_weight

    def _prioritize_pending_types(
        self,
        state: TaskState,
        all_types: List[str],
    ) -> List[str]:
        pending_types: List[str] = []
        if state.properties.get("type_area_needs_application", False):
            pending_types.extend(state.properties.get("type_area_pending_sources", []))
        if state.properties.get("object_type_needs_application", False):
            object_types = state.relations.get("object_type", {})
            pending_types.extend(
                object_types[obj]
                for obj in state.properties.get("object_type_pending_sources", [])
                if obj in object_types
            )
        pending_types = [
            object_type
            for object_type in dict.fromkeys(pending_types)
            if object_type in all_types
        ]
        if not pending_types:
            return all_types
        rest = [object_type for object_type in all_types if object_type not in pending_types]
        random.shuffle(pending_types)
        random.shuffle(rest)
        return pending_types + rest

    def sample_parameters(self, state: TaskState) -> CategoryCycleParameters:
        all_types = self._get_known_types(state)
        all_objects = state.attributes.get("objects", []).copy()
        all_areas = state.attributes.get("target_areas", [])
        if not all_types or not all_objects or not all_areas:
            raise RuntimeError(
                f"Failed to build the stage from {self.__class__.__name__} "
                "due to empty categories, objects or areas"
            )
        nb_categories = random.randint(1, min(self.max_categories, len(all_types)))
        selected_types = self._prioritize_pending_types(state, all_types)[:nb_categories]
        object_types = state.relations.get("object_type", {})
        type_areas = state.relations.get("type_area", {})
        type_to_objects = {
            object_type: [
                object_name
                for object_name, associated_type in object_types.items()
                if associated_type == object_type and object_name in all_objects
            ]
            for object_type in selected_types
        }
        if not any(type_to_objects.values()):
            raise RuntimeError(
                f"The {self.__class__.__name__} failed to build a cycle because "
                "no object could be associated to the selected categories"
            )
        missing_type_assignment: Dict[str, str] = {}
        assignment: Dict[str, str] = {}
        default_area = state.properties.get("relation_default_targets", {}).get(
            "type_area", "none"
        )
        for object_type, objects in type_to_objects.items():
            target_area = type_areas.get(object_type, default_area)
            if target_area not in all_areas:
                target_area = random.choice(all_areas)
                missing_type_assignment[object_type] = target_area
            for object_name in objects:
                assignment[object_name] = target_area

        if len(selected_types) == 1:
            initial_instruction = (
                "Launch a cycle for all objects from category "
                f"{selected_types[0]}."
            )
        else:
            initial_instruction = (
                "Launch a cycle for all objects from categories "
                f"{' and '.join(selected_types)}."
            )
        question = None
        resolution_input = initial_instruction
        if missing_type_assignment:
            missing_types = " and ".join(missing_type_assignment)
            question = f"Which target areas should be used for {missing_types}?"
            resolution_input = ". ".join(
                f"For this cycle, consider that category {object_type} goes to {area}"
                for object_type, area in missing_type_assignment.items()
            ) + "."
        clarification = (
            CategoryClarification(
                tuple(missing_type_assignment.items()),
                question,
                resolution_input,
            )
            if question is not None
            else None
        )
        return CategoryCycleParameters(
            tuple(selected_types),
            tuple(all_areas),
            tuple(assignment.items()),
            initial_instruction,
            clarification,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: CategoryCycleParameters,
    ) -> list[BaseTaskStage]:
        stages: list[BaseTaskStage] = []
        instruction = UserInstruction(parameters.initial_instruction)
        if parameters.clarification is not None:
            missing_types = " and ".join(
                object_type
                for object_type, _ in parameters.clarification.missing_assignment
            )
            stages.append(
                MissingInformationStage(
                    instruction=instruction,
                    answer=f"The robot must ask about target areas for {missing_types}",
                )
            )
            instruction = UserInstruction(parameters.clarification.answer)
        for stage in build_object_to_zone_stages(
            assignment=dict(parameters.assignment),
            known_areas=list(parameters.all_areas),
            flag_answer=True,
            instruction=instruction,
            final_message_decision=True,
        ):
            if stages:
                stage.stage_input.linked_to_prev = True
            stages.append(stage)
        return stages


    def apply_request(
        self,
        state: TaskState,
        parameters: CategoryCycleParameters,
    ) -> TaskState:
        if state.properties.get("object_type_needs_application", False):
            state.properties["object_type_needs_application"] = False
            state.properties.pop("object_type_pending_sources", None)
        if state.properties.get("type_area_needs_application", False):
            state.properties["type_area_needs_application"] = False
            state.properties.pop("type_area_pending_sources", None)
        state.properties["category_cycle_applications"] = (
            state.properties.get("category_cycle_applications", 0) + 1
        )
        return state

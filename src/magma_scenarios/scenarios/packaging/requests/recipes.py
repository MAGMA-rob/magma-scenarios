from dataclasses import dataclass
import random
from typing import List, Tuple

from magma_core.simulation.data_structures import UserInstruction
from magma_core.simulation.requests import BaseRequest
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.utils.text_utils import join_with_and
from magma_scenarios.templates.stages import ForbiddenElemStage

from ..environment_transitions import ResetPackagingTransition
from ..packaging_constraints import (
    DEFAULT_RECIPE_KEY,
    DEFAULT_RECIPE_PENDING_KEY,
    INCOMPATIBILITY_RULES_PENDING_KEY,
)
from ..packaging_stages import (
    PackagingCompositionProgressStage,
    PackagingCompositionStage,
)
from ..recipe_engine import RecipeEngine


@dataclass(frozen=True)
class PrepareTrayParameters:
    composition: Tuple[str, ...]
    instruction: str


def _progress_stages(
    composition: Tuple[str, ...],
    instruction: str,
    stop_after: int,
) -> List[BaseTaskStage]:
    return [
        PackagingCompositionProgressStage(
            expected_objects=composition,
            minimum=minimum,
            instruction=instruction if minimum == 1 else "none",
            entry_transition=(
                ResetPackagingTransition() if minimum == 1 else None
            ),
        )
        for minimum in range(1, stop_after + 1)
    ]


class PrepareTrayRequest(BaseRequest[PrepareTrayParameters]):
    """Prepare either the default recipe or one explicit valid recipe."""

    def sampling_weight(self, state: TaskState) -> float:
        if state.properties.get(DEFAULT_RECIPE_PENDING_KEY, False):
            return 8.0
        if state.properties.get(INCOMPATIBILITY_RULES_PENDING_KEY, False):
            return 6.0
        return 4.0

    def sample_parameters(self, state: TaskState) -> PrepareTrayParameters:
        engine = RecipeEngine(state)
        default_recipe = tuple(state.properties.get(DEFAULT_RECIPE_KEY, []))
        if default_recipe:
            validation = engine.validate(default_recipe)
            if not validation.valid:
                raise RuntimeError(
                    "The default recipe is invalid: " + validation.description()
                )
            replacement_candidates = engine.valid_replacement_plans(
                default_recipe
            )
            explicit_candidates = engine.valid_compositions(
                excluded=[default_recipe]
            )
            modes = ["default"]
            weights = [2]
            if replacement_candidates:
                modes.append("replacement")
                weights.append(5)
            if explicit_candidates:
                modes.append("explicit")
                weights.append(2)

            if state.properties.get(DEFAULT_RECIPE_PENDING_KEY, False):
                pending_weights = {
                    "default": 4,
                    "replacement": 4,
                    "explicit": 1,
                }
                weights = [pending_weights[mode] for mode in modes]
            mode = random.choices(modes, weights=weights, k=1)[0]

            if mode == "default":
                instruction = random.choice((
                    "Hello, I would like a tray.",
                    "Please prepare the standard tray.",
                    "Can you make me a regular tray?",
                ))
                return PrepareTrayParameters(default_recipe, instruction)

            if mode == "replacement":
                replacement = random.choice(replacement_candidates)
                instruction = random.choice((
                    "Prepare the standard tray, but replace "
                    f"{replacement.removed} with {replacement.added}.",
                    "I would like the default recipe with "
                    f"{replacement.added} instead of {replacement.removed}.",
                ))
                return PrepareTrayParameters(
                    replacement.result_recipe,
                    instruction,
                )

            composition = engine.sample_valid_composition(
                excluded=[default_recipe]
            )
            instruction = random.choice((
                f"Please prepare a tray with {join_with_and(list(composition))}.",
                f"I would like a tray containing {join_with_and(list(composition))}.",
            ))
            return PrepareTrayParameters(composition, instruction)

        composition = engine.sample_valid_composition()
        instruction = random.choice((
            f"Please prepare a tray with {join_with_and(list(composition))}.",
            f"I would like a tray containing {join_with_and(list(composition))}.",
        ))
        return PrepareTrayParameters(composition, instruction)

    def create_stages(
        self,
        state: TaskState,
        parameters: PrepareTrayParameters,
    ) -> List[BaseTaskStage]:
        return [
            PackagingCompositionStage(
                expected_objects=parameters.composition,
                instruction=parameters.instruction,
                entry_transition=ResetPackagingTransition(),
            )
        ]

    def apply_request(
        self,
        state: TaskState,
        parameters: PrepareTrayParameters,
    ) -> TaskState:
        state.properties[DEFAULT_RECIPE_PENDING_KEY] = False
        state.properties[INCOMPATIBILITY_RULES_PENDING_KEY] = False
        return state


@dataclass(frozen=True)
class PackagingInterruptionParameters:
    initial_composition: Tuple[str, ...]
    final_composition: Tuple[str, ...]
    removed: str
    added: str
    interrupt_after: int
    initial_instruction: str
    interruption_instruction: str


class PackagingInterruptionRequest(
    BaseRequest[PackagingInterruptionParameters]
):
    """Replace one requested food while a valid tray is being prepared."""

    def sampling_weight(self, state: TaskState) -> float:
        engine = RecipeEngine(state)
        default_recipe = tuple(state.properties.get(DEFAULT_RECIPE_KEY, []))
        if default_recipe:
            return 2.0 if engine.valid_replacement_plans(default_recipe) else 0.0
        return 2.0 if any(
            engine.valid_replacement_plans(composition)
            for composition in engine.valid_compositions()
        ) else 0.0

    def parameters_for_prepare(
        self,
        state: TaskState,
        parameters: PrepareTrayParameters,
    ) -> PackagingInterruptionParameters:
        engine = RecipeEngine(state)
        replacement_plans = engine.valid_replacement_plans(
            parameters.composition
        )
        if not replacement_plans:
            raise RuntimeError(
                "The requested tray has no constraint-compatible replacement"
            )
        replacement = random.choice(replacement_plans)
        interrupt_after = 1
        interruption_instruction = random.choice((
            f"Actually, replace {replacement.removed} with {replacement.added} "
            "on this tray.",
            f"I changed my mind: use {replacement.added} instead of "
            f"{replacement.removed} for this tray.",
        ))
        return PackagingInterruptionParameters(
            initial_composition=parameters.composition,
            final_composition=replacement.result_recipe,
            removed=replacement.removed,
            added=replacement.added,
            interrupt_after=interrupt_after,
            initial_instruction=parameters.instruction,
            interruption_instruction=interruption_instruction,
        )

    def sample_parameters(
        self,
        state: TaskState,
    ) -> PackagingInterruptionParameters:
        prepare_parameters = PrepareTrayRequest().sample_parameters(state)
        return self.parameters_for_prepare(state, prepare_parameters)

    def create_stages(
        self,
        state: TaskState,
        parameters: PackagingInterruptionParameters,
    ) -> List[BaseTaskStage]:
        stages = _progress_stages(
            parameters.initial_composition,
            parameters.initial_instruction,
            parameters.interrupt_after,
        )
        stages.append(
            PackagingCompositionStage(
                expected_objects=parameters.final_composition,
                instruction=parameters.interruption_instruction,
                linked_to_prev=True,
                placed_object_count=parameters.interrupt_after,
                replacement_may_be_placed=True,
            )
        )
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: PackagingInterruptionParameters,
    ) -> TaskState:
        state.properties[DEFAULT_RECIPE_PENDING_KEY] = False
        state.properties[INCOMPATIBILITY_RULES_PENDING_KEY] = False
        return state


@dataclass(frozen=True)
class ReplaceTrayItemParameters:
    base_recipe: Tuple[str, ...]
    removed: str
    invalid_added: str
    invalid_recipe: Tuple[str, ...]
    violated_pair: Tuple[str, str]
    initial_instruction: str
    final_recipe: Tuple[str, ...]
    follow_up_instruction: str


class ReplaceTrayItemRequest(BaseRequest[ReplaceTrayItemParameters]):
    """Refuse one invalid replacement, then execute a valid correction."""

    def sampling_weight(self, state: TaskState) -> float:
        engine = RecipeEngine(state)
        default_recipe = state.properties.get(DEFAULT_RECIPE_KEY, [])
        plans = engine.incompatibility_violation_plans(default_recipe or None)
        if not plans:
            return 0.0
        if state.properties.get(INCOMPATIBILITY_RULES_PENDING_KEY, False):
            return 12.0
        return 8.0

    def sample_parameters(self, state: TaskState) -> ReplaceTrayItemParameters:
        engine = RecipeEngine(state)
        default_recipe = tuple(state.properties.get(DEFAULT_RECIPE_KEY, []))
        invalid_plan = engine.sample_incompatibility_violation(
            default_recipe or None
        )

        if default_recipe:
            initial_instruction = random.choice((
                "Prepare the default tray, but replace "
                f"{invalid_plan.removed} with {invalid_plan.added}.",
                "For the standard recipe, use "
                f"{invalid_plan.added} instead of {invalid_plan.removed}.",
            ))
        else:
            initial_instruction = (
                "Prepare a tray based on "
                f"{join_with_and(list(invalid_plan.base_recipe))}, but replace "
                f"{invalid_plan.removed} with {invalid_plan.added}."
            )

        use_default = bool(default_recipe) and random.choice([True, False])
        if use_default:
            final_recipe = default_recipe
            follow_up_instruction = random.choice((
                "Okay, then prepare the default recipe.",
                "I understand. Just make the standard tray instead.",
            ))
        else:
            correction = random.choice(
                engine.compatible_replacements(
                    invalid_plan.base_recipe,
                    invalid_plan.removed,
                )
            )
            final_recipe = correction.result_recipe
            follow_up_instruction = random.choice((
                f"Okay, replace {correction.removed} with {correction.added} instead.",
                f"Then use {correction.added} instead of {correction.removed}.",
            ))

        return ReplaceTrayItemParameters(
            base_recipe=invalid_plan.base_recipe,
            removed=invalid_plan.removed,
            invalid_added=invalid_plan.added,
            invalid_recipe=invalid_plan.result_recipe,
            violated_pair=invalid_plan.violated_pair or ("", ""),
            initial_instruction=initial_instruction,
            final_recipe=final_recipe,
            follow_up_instruction=follow_up_instruction,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: ReplaceTrayItemParameters,
    ) -> List[BaseTaskStage]:
        first, second = parameters.violated_pair
        refusal = ForbiddenElemStage(
            UserInstruction(parameters.initial_instruction),
            "The model must refuse because "
            f"{first} and {second} cannot be placed on the same tray.",
        )
        corrected_tray = PackagingCompositionStage(
            expected_objects=parameters.final_recipe,
            instruction=parameters.follow_up_instruction,
            linked_to_prev=True,
            entry_transition=ResetPackagingTransition(),
        )
        return [refusal, corrected_tray]

    def apply_request(
        self,
        state: TaskState,
        parameters: ReplaceTrayItemParameters,
    ) -> TaskState:
        state.properties[DEFAULT_RECIPE_PENDING_KEY] = False
        state.properties[INCOMPATIBILITY_RULES_PENDING_KEY] = False
        return state

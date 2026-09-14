import random
from typing import Optional

from magma_scenarios.templates.requests.interact_request import (
    BaseConstraintRequest,
    ConstraintParameters,
)
from magma_core.simulation.state import TaskState
from magma_core.utils.text_utils import join_with_and

from ..packaging_constraints import (
    DEFAULT_RECIPE_KEY,
    DEFAULT_RECIPE_PENDING_KEY,
    INCOMPATIBILITY_RULES_PENDING_KEY,
    MAX_ACTIVE_INCOMPATIBILITY_RULES,
    DefaultRecipeConstraint,
    IncompatibleFoodsConstraint,
)
from ..recipe_engine import RecipeEngine


class UpdateDefaultRecipeRequest(BaseConstraintRequest):
    """Define or modify the default tray recipe."""

    def sampling_weight(self, state: TaskState) -> float:
        engine = RecipeEngine(state)
        current_recipe = state.properties.get(DEFAULT_RECIPE_KEY, [])
        if not current_recipe:
            return 3.0 if engine.valid_compositions() else 0.0

        has_override = bool(engine.valid_compositions(excluded=[current_recipe]))
        has_replacement = bool(engine.valid_replacement_plans(current_recipe))
        return 1.0 if has_override or has_replacement else 0.0

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        engine = RecipeEngine(state)
        current_recipe = list(state.properties.get(DEFAULT_RECIPE_KEY, []))

        if not current_recipe:
            recipe = engine.sample_valid_composition()
            instruction = random.choice((
                f"The default tray recipe is {join_with_and(list(recipe))}.",
                "From now on, prepare the standard tray with "
                f"{join_with_and(list(recipe))}.",
            ))
            return ConstraintParameters(
                [DefaultRecipeConstraint(recipe)],
                instruction,
            )

        override_candidates = engine.valid_compositions(excluded=[current_recipe])
        replacement_candidates = engine.valid_replacement_plans(current_recipe)
        if not override_candidates and not replacement_candidates:
            raise RuntimeError("The current default recipe cannot be updated")

        if replacement_candidates and override_candidates:
            mode = random.choices(
                ("replacement", "override"),
                weights=(3, 1),
                k=1,
            )[0]
        elif replacement_candidates:
            mode = "replacement"
        else:
            mode = "override"

        if mode == "override":
            recipe = engine.sample_valid_composition(excluded=[current_recipe])
            instruction = random.choice((
                "Forget the previous default recipe. The new recipe is "
                f"{join_with_and(list(recipe))}.",
                "Replace the standard tray recipe with "
                f"{join_with_and(list(recipe))}.",
            ))
        else:
            plan = random.choice(replacement_candidates)
            recipe = plan.result_recipe
            instruction = random.choice((
                f"In the default recipe, replace {plan.removed} with {plan.added}.",
                f"The standard tray now uses {plan.added} instead of {plan.removed}.",
            ))

        return ConstraintParameters(
            [DefaultRecipeConstraint(recipe)],
            instruction,
        )

    def apply_request(
        self,
        state: TaskState,
        parameters: ConstraintParameters,
    ) -> TaskState:
        state = super().apply_request(state, parameters)
        state.properties[DEFAULT_RECIPE_PENDING_KEY] = True
        return state


class UpdateIncompatibilityRuleRequest(BaseConstraintRequest):
    """Add or replace one persistent two-food incompatibility rule."""

    def __init__(
        self,
        sampling_weight: float = 0.25,
        aged_sampling_weight: float = 4.0,
        requests_to_aged_weight: int = 3,
        steps_to_aged_weight: Optional[int] = None,
    ) -> None:
        super().__init__()
        if sampling_weight < 0 or aged_sampling_weight < 0:
            raise ValueError("Sampling weights must be non-negative")
        if requests_to_aged_weight <= 0:
            raise ValueError("requests_to_aged_weight must be positive")
        if steps_to_aged_weight is None:
            steps_to_aged_weight = requests_to_aged_weight
        if steps_to_aged_weight <= 0:
            raise ValueError("steps_to_aged_weight must be positive")

        self.base_sampling_weight = sampling_weight
        self.aged_sampling_weight = aged_sampling_weight
        self.requests_to_aged_weight = requests_to_aged_weight
        self.steps_to_aged_weight = steps_to_aged_weight

    def sampling_weight(self, state: TaskState) -> float:
        if not RecipeEngine(state).compatible_rule_pairs():
            return 0.0

        step_index = state.properties.get("_generator_step_index")
        if step_index is not None:
            last_index = state.properties.get(
                "packaging_rule_last_step_index",
                0,
            )
            age = max(0, step_index - last_index)
            age_to_max_weight = self.steps_to_aged_weight
        else:
            request_index = state.properties.get("_generator_request_index")
            if request_index is None:
                return self.base_sampling_weight
            last_index = state.properties.get(
                "packaging_rule_last_request_index",
                0,
            )
            age = max(0, request_index - last_index)
            age_to_max_weight = self.requests_to_aged_weight

        ratio = min(1.0, age / age_to_max_weight)
        return self.base_sampling_weight + (
            self.aged_sampling_weight - self.base_sampling_weight
        ) * ratio

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        engine = RecipeEngine(state)
        candidates = engine.compatible_rule_pairs()
        if not candidates:
            raise RuntimeError("No packaging incompatibility rule can be sampled")

        default_recipe = set(engine.default_recipe)
        replacement_friendly_candidates = [
            pair
            for pair in candidates
            if default_recipe and len(default_recipe.intersection(pair)) == 1
        ]
        new_pair = random.choice(replacement_friendly_candidates or candidates)
        active_rules = list(engine.incompatibility_rules)

        if len(active_rules) < MAX_ACTIVE_INCOMPATIBILITY_RULES:
            instruction = random.choice((
                f"Do not place {new_pair[0]} and {new_pair[1]} on the same tray.",
                f"A tray must never contain both {new_pair[0]} and {new_pair[1]}.",
            ))
            constraint = IncompatibleFoodsConstraint(*new_pair)
        else:
            replaced_pair = random.choice(active_rules)
            instruction = random.choice((
                "Replace the rule forbidding "
                f"{replaced_pair[0]} and {replaced_pair[1]} together with a rule "
                f"forbidding {new_pair[0]} and {new_pair[1]} together.",
                "The old rule saying that "
                f"{replaced_pair[0]} and {replaced_pair[1]} cannot share a tray "
                "is replaced: now "
                f"{new_pair[0]} and {new_pair[1]} cannot share a tray.",
            ))
            constraint = IncompatibleFoodsConstraint(
                *new_pair,
                replaced_pair=replaced_pair,
            )

        return ConstraintParameters([constraint], instruction)

    def apply_request(
        self,
        state: TaskState,
        parameters: ConstraintParameters,
    ) -> TaskState:
        state = super().apply_request(state, parameters)
        state.properties[INCOMPATIBILITY_RULES_PENDING_KEY] = True
        state.properties["packaging_rule_last_request_index"] = (
            state.properties.get("_generator_request_index", 0)
        )
        state.properties["packaging_rule_last_step_index"] = (
            state.properties.get("_generator_step_index", 0)
        )
        return state

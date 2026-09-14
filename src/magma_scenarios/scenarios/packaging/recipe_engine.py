from dataclasses import dataclass
from itertools import combinations
import random
from typing import Iterable, Optional, Sequence, Tuple

from magma_core.simulation.state import TaskState

from .packaging_constraints import (
    DEFAULT_RECIPE_KEY,
    INCOMPATIBILITY_RULES_KEY,
    canonical_food_pair,
)


MIN_RECIPE_SIZE = 2
MAX_RECIPE_SIZE = 4


@dataclass(frozen=True)
class RecipeValidation:
    valid: bool
    unknown_objects: Tuple[str, ...]
    duplicated_objects: Tuple[str, ...]
    incompatibility_violations: Tuple[Tuple[str, str], ...]
    invalid_size: bool

    def description(self) -> str:
        reasons = []
        if self.invalid_size:
            reasons.append(
                f"recipe size must be between {MIN_RECIPE_SIZE} and {MAX_RECIPE_SIZE}"
            )
        if self.unknown_objects:
            reasons.append(f"unknown objects: {', '.join(self.unknown_objects)}")
        if self.duplicated_objects:
            reasons.append(
                f"duplicated objects: {', '.join(self.duplicated_objects)}"
            )
        for first, second in self.incompatibility_violations:
            reasons.append(f"{first} and {second} cannot be placed together")
        return "; ".join(reasons) if reasons else "valid recipe"


@dataclass(frozen=True)
class ReplacementPlan:
    base_recipe: Tuple[str, ...]
    removed: str
    added: str
    result_recipe: Tuple[str, ...]
    violated_pair: Optional[Tuple[str, str]] = None


class RecipeEngine:
    """Validate and sample tray recipes from one immutable state snapshot."""

    def __init__(self, state: TaskState) -> None:
        self.object_type = dict(state.relations.get("object_type", {}))
        self.objects = list(self.object_type)
        self.incompatibility_rules = [
            canonical_food_pair(*pair)
            for pair in state.properties.get(INCOMPATIBILITY_RULES_KEY, [])
        ]
        self.default_recipe = list(state.properties.get(DEFAULT_RECIPE_KEY, []))

    def validate(self, composition: Sequence[str]) -> RecipeValidation:
        normalized = list(composition)
        unknown_objects = tuple(
            dict.fromkeys(food for food in normalized if food not in self.object_type)
        )
        duplicated_objects = tuple(
            food
            for food in dict.fromkeys(normalized)
            if normalized.count(food) > 1
        )
        composition_set = set(normalized)
        incompatibility_violations = tuple(
            pair
            for pair in self.incompatibility_rules
            if composition_set.issuperset(pair)
        )
        invalid_size = not MIN_RECIPE_SIZE <= len(normalized) <= MAX_RECIPE_SIZE

        return RecipeValidation(
            valid=(
                not invalid_size
                and not unknown_objects
                and not duplicated_objects
                and not incompatibility_violations
            ),
            unknown_objects=unknown_objects,
            duplicated_objects=duplicated_objects,
            incompatibility_violations=incompatibility_violations,
            invalid_size=invalid_size,
        )

    def valid_compositions(
        self,
        minimum: int = MIN_RECIPE_SIZE,
        maximum: int = MAX_RECIPE_SIZE,
        excluded: Iterable[Sequence[str]] = (),
    ) -> list[Tuple[str, ...]]:
        if minimum < MIN_RECIPE_SIZE or maximum > MAX_RECIPE_SIZE:
            raise ValueError(
                f"Recipe bounds must stay within {MIN_RECIPE_SIZE}..{MAX_RECIPE_SIZE}"
            )
        if minimum > maximum:
            raise ValueError("minimum cannot exceed maximum")

        excluded_sets = {frozenset(composition) for composition in excluded}
        candidates = []
        for size in range(minimum, maximum + 1):
            for composition in combinations(self.objects, size):
                if frozenset(composition) in excluded_sets:
                    continue
                if self.validate(composition).valid:
                    candidates.append(composition)
        return candidates

    def sample_valid_composition(
        self,
        minimum: int = MIN_RECIPE_SIZE,
        maximum: int = MAX_RECIPE_SIZE,
        excluded: Iterable[Sequence[str]] = (),
    ) -> Tuple[str, ...]:
        candidates = self.valid_compositions(minimum, maximum, excluded)
        if not candidates:
            raise RuntimeError("No valid packaging composition can be sampled")
        available_sizes = sorted({len(composition) for composition in candidates})
        selected_size = random.choice(available_sizes)
        return random.choice([
            composition
            for composition in candidates
            if len(composition) == selected_size
        ])

    def compatible_replacements(
        self,
        composition: Sequence[str],
        removed: str,
    ) -> list[ReplacementPlan]:
        base_recipe = tuple(composition)
        if removed not in base_recipe:
            raise ValueError(f"{removed!r} is not part of the recipe")

        replacement_index = base_recipe.index(removed)
        plans = []
        for candidate in self.objects:
            if candidate in base_recipe:
                continue
            result = list(base_recipe)
            result[replacement_index] = candidate
            if self.validate(result).valid:
                plans.append(
                    ReplacementPlan(base_recipe, removed, candidate, tuple(result))
                )
        return plans

    def valid_replacement_plans(
        self,
        composition: Sequence[str],
    ) -> list[ReplacementPlan]:
        plans = []
        for removed in composition:
            plans.extend(self.compatible_replacements(composition, removed))
        return plans

    def compatible_rule_pairs(self) -> list[Tuple[str, str]]:
        default_recipe = set(self.default_recipe)
        active_rules = set(self.incompatibility_rules)
        candidates = []
        for first, second in combinations(self.objects, 2):
            pair = canonical_food_pair(first, second)
            if pair in active_rules or default_recipe.issuperset(pair):
                continue
            candidates.append(pair)
        return candidates

    def incompatibility_violation_plans(
        self,
        composition: Optional[Sequence[str]] = None,
    ) -> list[ReplacementPlan]:
        if not self.incompatibility_rules:
            return []

        base_recipes = (
            [tuple(composition)]
            if composition is not None
            else self.valid_compositions()
        )
        plans = []
        for base_recipe in base_recipes:
            if not self.validate(base_recipe).valid:
                continue
            for removed in base_recipe:
                replacement_index = base_recipe.index(removed)
                for candidate in self.objects:
                    if candidate in base_recipe:
                        continue
                    result = list(base_recipe)
                    result[replacement_index] = candidate
                    validation = self.validate(result)
                    if len(validation.incompatibility_violations) != 1:
                        continue
                    if not self.compatible_replacements(base_recipe, removed):
                        continue
                    plans.append(
                        ReplacementPlan(
                            base_recipe=base_recipe,
                            removed=removed,
                            added=candidate,
                            result_recipe=tuple(result),
                            violated_pair=validation.incompatibility_violations[0],
                        )
                    )
        return plans

    def sample_incompatibility_violation(
        self,
        composition: Optional[Sequence[str]] = None,
    ) -> ReplacementPlan:
        plans = self.incompatibility_violation_plans(composition)
        if not plans:
            raise RuntimeError("No single-rule packaging violation can be sampled")
        return random.choice(plans)

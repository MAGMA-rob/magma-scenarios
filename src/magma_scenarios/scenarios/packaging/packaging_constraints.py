from typing import Optional, Sequence, Tuple

from magma_core.simulation.constraints import BaseConstraint
from magma_core.simulation.state import TaskState

from .attributes import FOOD_OBJECTS


DEFAULT_RECIPE_KEY = "default_recipe"
INCOMPATIBILITY_RULES_KEY = "incompatibility_rules"
DEFAULT_RECIPE_PENDING_KEY = "default_recipe_needs_application"
INCOMPATIBILITY_RULES_PENDING_KEY = "incompatibility_rules_need_application"
MAX_ACTIVE_INCOMPATIBILITY_RULES = 2


def canonical_food_pair(first: str, second: str) -> Tuple[str, str]:
    if first == second:
        raise ValueError("An incompatibility rule requires two different foods")
    unknown = {first, second} - set(FOOD_OBJECTS)
    if unknown:
        raise ValueError(f"Unknown food objects: {sorted(unknown)}")

    order = {food: index for index, food in enumerate(FOOD_OBJECTS)}
    return tuple(sorted((first, second), key=order.__getitem__))


class DefaultRecipeConstraint(BaseConstraint):
    def __init__(self, recipe: Sequence[str]) -> None:
        super().__init__()
        self.recipe = list(recipe)

    def apply(self, state: TaskState):
        from .recipe_engine import RecipeEngine

        validation = RecipeEngine(state).validate(self.recipe)
        if not validation.valid:
            raise RuntimeError(
                "The default recipe must be a valid tray composition: "
                f"{validation.description()}"
            )

        super().apply(state)
        state.properties[DEFAULT_RECIPE_KEY] = self.recipe.copy()

    def outdated(self, state: TaskState) -> bool:
        known_objects = state.relations.get("object_type", {})
        return any(food not in known_objects for food in self.recipe)


class IncompatibleFoodsConstraint(BaseConstraint):
    """Add one incompatibility rule, optionally replacing an existing one."""

    def __init__(
        self,
        first: str,
        second: str,
        replaced_pair: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.pair = canonical_food_pair(first, second)
        self.replaced_pair = (
            canonical_food_pair(*replaced_pair) if replaced_pair else None
        )

    def apply(self, state: TaskState):
        known_objects = state.relations.get("object_type", {})
        if any(food not in known_objects for food in self.pair):
            raise RuntimeError(f"Unknown food pair {self.pair!r}")

        rules = [
            canonical_food_pair(*pair)
            for pair in state.properties.get(INCOMPATIBILITY_RULES_KEY, [])
        ]
        if self.replaced_pair is not None:
            if self.replaced_pair not in rules:
                raise RuntimeError(
                    f"The rule {self.replaced_pair!r} cannot be replaced because "
                    "it is not active"
                )
            rules.remove(self.replaced_pair)

        if self.pair in rules:
            raise RuntimeError(f"The rule {self.pair!r} is already active")
        rules.append(self.pair)
        if len(rules) > MAX_ACTIVE_INCOMPATIBILITY_RULES:
            raise RuntimeError(
                "Packaging supports at most two active incompatibility rules"
            )

        default_recipe = set(state.properties.get(DEFAULT_RECIPE_KEY, []))
        if default_recipe.issuperset(self.pair):
            raise RuntimeError(
                "The incompatibility rule conflicts with the default recipe"
            )

        super().apply(state)
        state.properties[INCOMPATIBILITY_RULES_KEY] = rules

    def outdated(self, state: TaskState) -> bool:
        known_objects = state.relations.get("object_type", {})
        return any(food not in known_objects for food in self.pair)

from typing import Dict, List, Literal

from magma_core.base.constraints import BaseConstraint
from magma_core.base.state import TaskState

from magma_scenarios.templates.constraints import RelationAssignmentConstraint


COFFEE_PREFERENCE_KEY = "coffee_preference"
TEAM_COFFEE_RULES_KEY = "team_coffee_preference_rules"


def _get_coffee_preferences(state: TaskState) -> Dict[str, str]:
    if COFFEE_PREFERENCE_KEY not in state.relations:
        state.relations[COFFEE_PREFERENCE_KEY] = {}
    return state.relations[COFFEE_PREFERENCE_KEY]


def _get_team_rule_specs(state: TaskState) -> Dict[str, Dict]:
    if TEAM_COFFEE_RULES_KEY not in state.relations:
        state.relations[TEAM_COFFEE_RULES_KEY] = {}
    return state.relations[TEAM_COFFEE_RULES_KEY]


class CoffeePreferenceConstraint(RelationAssignmentConstraint):

    def __init__(
            self,
            name: str,
            coffee: str
        ) -> None:
        super().__init__(name, coffee, COFFEE_PREFERENCE_KEY, None, "coffee_pod")


class TeamCoffeePreferenceConstraint(BaseConstraint):

    def __init__(
            self,
            team_name: str,
            team_members: List[str],
            coffee: str,
            mode: Literal["override", "default"] = "override",
        ) -> None:
        super().__init__()
        if len(team_members) == 0:
            raise ValueError("team_members must not be empty")
        if mode not in ["override", "default"]:
            raise ValueError(f"Unsupported team coffee preference mode: {mode}")

        self.team_name = team_name
        self.team_members = list(dict.fromkeys(team_members))
        self.coffee = coffee
        self.mode = mode

    def apply(self, state: TaskState):
        super().apply(state)

        if self.coffee not in state.attributes.get("coffee_pod", []):
            raise RuntimeError(
                f"{self.coffee} is not a known coffee pod for {self.__class__.__name__}"
            )

        team_rule_specs = _get_team_rule_specs(state)
        team_rule_specs[self.team_name] = {
            "coffee": self.coffee,
            "mode": self.mode,
            "members": self.team_members.copy(),
        }

        preferences = _get_coffee_preferences(state)
        for member in self.team_members:
            if self.mode == "default" and member in preferences:
                continue
            preferences[member] = self.coffee

    def outdated(self, state: TaskState) -> bool:
        return self.coffee not in state.attributes.get("coffee_pod", [])


# Backward compatibility with the initial pluralized name.
CoffeePreferenceConstraints = CoffeePreferenceConstraint

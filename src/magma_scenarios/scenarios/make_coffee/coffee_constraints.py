from typing import Dict, List, Literal

from magma_core.simulation.constraints import BaseConstraint
from magma_core.simulation.state import TaskState

from magma_scenarios.templates.constraints import RelationAssignmentConstraint


COFFEE_PREFERENCE_KEY = "coffee_preference"
TEAM_COFFEE_RULES_KEY = "team_coffee_preference_rules"
UNAVAILABLE_COFFEE_PODS_KEY = "unavailable_coffee_pods"


def _get_coffee_preferences(state: TaskState) -> Dict[str, str]:
    if COFFEE_PREFERENCE_KEY not in state.relations:
        state.relations[COFFEE_PREFERENCE_KEY] = {}
    return state.relations[COFFEE_PREFERENCE_KEY]


def _get_team_rule_specs(state: TaskState) -> Dict[str, Dict]:
    if TEAM_COFFEE_RULES_KEY not in state.relations:
        state.relations[TEAM_COFFEE_RULES_KEY] = {}
    return state.relations[TEAM_COFFEE_RULES_KEY]


def get_unavailable_coffee_pods(state: TaskState) -> List[str]:
    """Return the currently unavailable pods that still exist in attributes."""
    known_pods = state.attributes.get("coffee_pod", [])
    return [
        pod
        for pod in state.properties.get(UNAVAILABLE_COFFEE_PODS_KEY, [])
        if pod in known_pods
    ]


def get_available_coffee_pods(state: TaskState) -> List[str]:
    unavailable_pods = set(get_unavailable_coffee_pods(state))
    return [
        pod
        for pod in state.attributes.get("coffee_pod", [])
        if pod not in unavailable_pods
    ]


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

    def outdated(self, state: TaskState) -> bool:
        return self.coffee not in state.attributes.get("coffee_pod", [])


class CoffeeUnavailableConstraint(BaseConstraint):
    """Persistently mark one coffee pod as unavailable."""

    def __init__(self, coffee: str) -> None:
        super().__init__()
        self.coffee = coffee

    def apply(self, state: TaskState):
        super().apply(state)
        if self.coffee not in state.attributes.get("coffee_pod", []):
            raise RuntimeError(
                f"{self.coffee} is not a known coffee pod for {self.__class__.__name__}"
            )
        state.properties[UNAVAILABLE_COFFEE_PODS_KEY] = [self.coffee]

    def outdated(self, state: TaskState) -> bool:
        return self.coffee not in state.attributes.get("coffee_pod", [])


class CoffeeAvailableConstraint(BaseConstraint):
    """Remove the current coffee pod unavailability."""

    def __init__(self, coffee: str) -> None:
        super().__init__()
        self.coffee = coffee

    def apply(self, state: TaskState):
        super().apply(state)
        state.properties[UNAVAILABLE_COFFEE_PODS_KEY] = []

    def outdated(self, state: TaskState) -> bool:
        return self.coffee not in state.attributes.get("coffee_pod", [])


# Backward compatibility with the initial pluralized name.
CoffeePreferenceConstraints = CoffeePreferenceConstraint

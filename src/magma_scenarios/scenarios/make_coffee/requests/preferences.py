import random
from collections import defaultdict
from typing import Dict, List, Literal

from magma_scenarios.templates.requests.interact_request import (
    BaseConstraintRequest,
    ConstraintParameters,
)
from magma_core.simulation.state import TaskState
from magma_core.utils.text_utils import join_with_and

from ..coffee_constraints import (
    CoffeePreferenceConstraint,
    TeamCoffeePreferenceConstraint,
)


class GiveCoffeePreference(BaseConstraintRequest):
    def __init__(
        self,
        possible_names: List[str],
        max_name: int = 4,
        max_different_name: int = 6,
    ) -> None:
        super().__init__()
        self.max_nb = max_name
        self.possible_names = possible_names
        self.max_difference = max_different_name

    def _eligible_coffees_by_name(
        self,
        state: TaskState,
    ) -> Dict[str, List[str]]:
        coffees = state.attributes.get("coffee_pod", [])
        preferences = state.relations.get("coffee_preference", {})
        if len(preferences) >= self.max_difference:
            candidate_names = list(preferences)
        else:
            candidate_names = self.possible_names

        team_preferences = {
            member: rule.get("coffee")
            for rule in state.relations.get(
                "team_coffee_preference_rules", {}
            ).values()
            for member in rule.get("members", [])
        }
        eligible: Dict[str, List[str]] = {}
        for name in candidate_names:
            current_preference = preferences.get(name)
            team_preference = team_preferences.get(name)
            current_rendered_preference = (
                None
                if current_preference == team_preference
                else current_preference
            )
            candidate_coffees = [
                coffee
                for coffee in coffees
                if (
                    None if coffee == team_preference else coffee
                ) != current_rendered_preference
            ]
            if candidate_coffees:
                eligible[name] = candidate_coffees
        return eligible

    def sampling_weight(self, state: TaskState) -> float:
        if not self._eligible_coffees_by_name(state):
            return 0
        if state.properties.get("coffee_preference_needs_application", False):
            return 0.25
        if len(state.relations.get("coffee_preference", {})) < 1:
            return 4
        if len(state.relations.get("coffee_preference", {})) >= self.max_difference:
            return 0.5
        return 1

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        eligible_coffees = self._eligible_coffees_by_name(state)
        if not eligible_coffees:
            raise RuntimeError("No coffee preference can be changed")

        preferences = state.relations.get("coffee_preference", {})
        maximum_names = min(len(eligible_coffees), self.max_nb)
        if len(preferences) < self.max_difference:
            maximum_names = min(
                maximum_names,
                self.max_difference - len(preferences),
            )
        n = random.randint(1, maximum_names)
        names = random.sample(list(eligible_coffees), k=n)
        coffees = [
            random.choice(eligible_coffees[name])
            for name in names
        ]

        assignment = defaultdict(list)
        constraints = []
        for name, pod in zip(names, coffees):
            assignment[pod].append(name)
            constraints.append(CoffeePreferenceConstraint(name, pod))
        parts = [
            f"{join_with_and(name_list)} like their coffee {pod}"
            for pod, name_list in assignment.items()
        ]
        return ConstraintParameters(
            constraints,
            "Hey, please remember that " + ", ".join(parts) + ".",
        )

    def apply_request(
        self,
        state: TaskState,
        parameters: ConstraintParameters,
    ) -> TaskState:
        state = super().apply_request(state, parameters)
        state.properties["coffee_preference_needs_application"] = True
        return state


class GiveTeamCoffeePreference(BaseConstraintRequest):
    def __init__(
        self,
        team_assignment: Dict[str, List[str]],
        max_team_assignment: int = 2,
        mode: Literal["override", "default"] = "override",
    ) -> None:
        super().__init__()
        self.team_assignment = team_assignment
        self.max_team_assignment = max_team_assignment
        self.mode = mode

    def _get_eligible_teams(self, state: TaskState) -> List[str]:
        coffees = state.attributes.get("coffee_pod", [])
        if self.mode == "override":
            candidate_teams = [
                team_name
                for team_name, members in self.team_assignment.items()
                if len(members) > 0
            ]
        else:
            preferences = state.relations.get("coffee_preference", {})
            candidate_teams = [
                team_name
                for team_name, members in self.team_assignment.items()
                if len(members) > 0
                and any(member not in preferences for member in members)
            ]

        current_rules = state.relations.get(
            "team_coffee_preference_rules", {}
        )
        eligible_teams = []
        for team_name in candidate_teams:
            current_rule = current_rules.get(team_name)
            if any(
                current_rule is None
                or current_rule.get("coffee") != coffee
                or current_rule.get("mode") != self.mode
                for coffee in coffees
            ):
                eligible_teams.append(team_name)
        return eligible_teams

    def _eligible_coffees(
        self,
        state: TaskState,
        team_name: str,
    ) -> List[str]:
        current_rule = state.relations.get(
            "team_coffee_preference_rules", {}
        ).get(team_name)
        return [
            coffee
            for coffee in state.attributes.get("coffee_pod", [])
            if current_rule is None
            or current_rule.get("coffee") != coffee
            or current_rule.get("mode") != self.mode
        ]

    def sampling_weight(self, state: TaskState) -> float:
        if len(state.attributes.get("coffee_pod", [])) == 0:
            return 0
        if len(self._get_eligible_teams(state)) == 0:
            return 0
        if state.properties.get("team_coffee_preference_needs_application", False):
            return 0.25
        if len(state.relations.get("team_coffee_preference_rules", {})) < 1:
            return 3
        return 1

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        eligible_teams = self._get_eligible_teams(state)
        if len(eligible_teams) == 0:
            raise RuntimeError(
                f"Failed to sample a team in {self.__class__.__name__}"
            )

        coffees = state.attributes.get("coffee_pod", [])
        if len(coffees) == 0:
            raise RuntimeError("Empty coffee pod")

        nb_team = random.randint(
            1,
            min(self.max_team_assignment, len(eligible_teams)),
        )
        selected_teams = random.sample(eligible_teams, k=nb_team)
        selected_coffees = [
            random.choice(self._eligible_coffees(state, team_name))
            for team_name in selected_teams
        ]

        constraints = []
        msg_parts = []
        benchmark_like_msg_parts = []
        for team_name, coffee in zip(selected_teams, selected_coffees):
            constraints.append(
                TeamCoffeePreferenceConstraint(
                    team_name=team_name,
                    team_members=self.team_assignment[team_name],
                    coffee=coffee,
                    mode=self.mode,
                )
            )
            if self.mode == "override":
                msg_parts.append(f"team {team_name} prefers {coffee} coffee")
                benchmark_like_msg_parts.append(
                    f"team {team_name} prefers {coffee} coffee"
                )
            else:
                msg_parts.append(
                    f"team {team_name} prefers {coffee} coffee by default"
                )
                benchmark_like_msg_parts.append(
                    f"team {team_name} prefers {coffee} coffee by default"
                )

        instruction = (
            "Hello, please remember that " + ", and ".join(msg_parts) + "."
            if random.choice([True, False])
            else " and ".join(benchmark_like_msg_parts) + "."
        )
        return ConstraintParameters(constraints, instruction)

    def apply_request(
        self,
        state: TaskState,
        parameters: ConstraintParameters,
    ) -> TaskState:
        state = super().apply_request(state, parameters)
        state.properties["team_coffee_preference_needs_application"] = True
        return state

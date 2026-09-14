from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple

from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_core.utils.text_utils import is_or_are, join_with_and

from ..coffee_stages import (
    AskCoffeePreferenceForUserStage,
    AskPeopleInTeamStage,
    AskTeamCoffeePreferencesStage,
    AskTeamsForPeopleStage,
)


@dataclass(frozen=True)
class PeopleTeamQueryParameters:
    names: Tuple[str, ...]
    teams: Tuple[str, ...]
    instruction: str
    answer: str


@dataclass(frozen=True)
class TeamPeopleQueryParameters:
    team_name: str
    members: Tuple[str, ...]
    instruction: str
    answer: str


@dataclass(frozen=True)
class PreferenceQueryParameters:
    person: str
    coffee: str
    instruction: str
    answer: str


@dataclass(frozen=True)
class TeamPreferenceQueryParameters:
    team_name: str
    members: Tuple[str, ...]
    known_preferences: Tuple[Tuple[str, str], ...]
    focus: str
    instruction: str
    answer: str


class AskPeopleTeam(BaseRequest[PeopleTeamQueryParameters]):
    def __init__(
        self,
        team_assignment: Dict[str, List[str]],
        max_nb: int = 2,
    ) -> None:
        super().__init__()
        self.max_nb = max_nb
        self.people_to_team: Dict[str, str] = {}
        for team_name, members in team_assignment.items():
            for member in members:
                if member in self.people_to_team:
                    raise ValueError(f"{member} is assigned to multiple teams")
                self.people_to_team[member] = team_name

    def sampling_weight(self, state: TaskState) -> float:
        if not self.people_to_team:
            return 0
        if state.properties.get(
            "team_coffee_preference_needs_application", False
        ):
            return 0.75
        return 1

    def sample_parameters(self, state: TaskState) -> PeopleTeamQueryParameters:
        if not self.people_to_team:
            raise RuntimeError("The team assignment is empty")
        count = random.randint(1, min(self.max_nb, len(self.people_to_team)))
        names = random.sample(list(self.people_to_team), count)
        teams = [self.people_to_team[name] for name in names]
        instruction = (
            f"What team does {names[0]} belong to?"
            if len(names) == 1
            else f"In which teams are {join_with_and(names)}?"
        )
        answer = join_with_and([
            f"{person} is in team {team}"
            for person, team in zip(names, teams)
        ]) + "."
        return PeopleTeamQueryParameters(
            tuple(names), tuple(teams), instruction, answer
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: PeopleTeamQueryParameters,
    ) -> List[BaseTaskStage]:
        return [AskTeamsForPeopleStage(list(parameters.names), list(parameters.teams))]


class AskPeopleInTeam(BaseRequest[TeamPeopleQueryParameters]):
    def __init__(self, team_assignment: Dict[str, List[str]]) -> None:
        super().__init__()
        self.team_assignment = team_assignment

    def sampling_weight(self, state: TaskState) -> float:
        if not any(self.team_assignment.values()):
            return 0
        if state.properties.get(
            "team_coffee_preference_needs_application", False
        ):
            return 1.5
        return 1

    def sample_parameters(self, state: TaskState) -> TeamPeopleQueryParameters:
        non_empty_teams = [
            name for name, members in self.team_assignment.items() if members
        ]
        if not non_empty_teams:
            raise RuntimeError("The team assignment is empty")
        team_name = random.choice(non_empty_teams)
        members = self.team_assignment[team_name]
        instruction = f"Who is in team {team_name}?"
        answer = (
            f"{join_with_and(members)} {is_or_are(members)} in team {team_name}."
        )
        return TeamPeopleQueryParameters(
            team_name, tuple(members), instruction, answer
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: TeamPeopleQueryParameters,
    ) -> List[BaseTaskStage]:
        return [AskPeopleInTeamStage(parameters.team_name, list(parameters.members))]


class AskCoffeePreferenceForUser(BaseRequest[PreferenceQueryParameters]):
    def __init__(self, possible_names: List[str]) -> None:
        super().__init__()
        self.possible_names = possible_names

    def _known_people(self, state: TaskState) -> List[str]:
        preferences = state.relations.get("coffee_preference", {})
        return [name for name in self.possible_names if name in preferences]

    def sampling_weight(self, state: TaskState) -> float:
        if not self._known_people(state):
            return 0
        if state.properties.get("coffee_preference_needs_application", False):
            return 2
        if state.properties.get(
            "team_coffee_preference_needs_application", False
        ):
            return 1.5
        return 1

    def sample_parameters(self, state: TaskState) -> PreferenceQueryParameters:
        preferences = state.relations.get("coffee_preference", {})
        known_people = self._known_people(state)
        if not known_people:
            raise RuntimeError("No coffee preference is known")
        person = random.choice(known_people)
        question = random.choice([
            f"What coffee does {person} like?",
            f"What is {person}'s coffee preference?",
            f"Which coffee like {person}?",
        ])
        coffee = preferences[person]
        return PreferenceQueryParameters(
            person,
            coffee,
            question,
            f"{person} likes {coffee} coffee.",
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: PreferenceQueryParameters,
    ) -> List[BaseTaskStage]:
        return [
            AskCoffeePreferenceForUserStage(
                parameters.person,
                parameters.coffee,
                question=parameters.instruction,
            )
        ]


class AskCoffeePreferenceInTeam(BaseRequest[TeamPreferenceQueryParameters]):
    def __init__(
        self,
        team_assignment: Dict[str, List[str]],
        allowed_focus: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.team_assignment = team_assignment
        self.allowed_focus = allowed_focus

    def _get_available_focus(
        self,
        state: TaskState,
        team_members: List[str],
    ) -> List[str]:
        preferences = state.relations.get("coffee_preference", {})
        coffee_pods = state.attributes.get("coffee_pod", [])
        candidate_focus = (
            ["all", *coffee_pods, "unknown"]
            if self.allowed_focus is None
            else self.allowed_focus
        )
        available_focus = []
        for focus in candidate_focus:
            if focus == "all":
                available_focus.append(focus)
            elif focus == "unknown" and any(
                name not in preferences for name in team_members
            ):
                available_focus.append(focus)
            elif any(preferences.get(name) == focus for name in team_members):
                available_focus.append(focus)
        return available_focus or ["all"]

    def sampling_weight(self, state: TaskState) -> float:
        if not any(self.team_assignment.values()):
            return 0
        if state.properties.get(
            "team_coffee_preference_needs_application", False
        ):
            return 6
        if state.relations.get("coffee_preference", {}):
            return 3
        return 1

    def sample_parameters(self, state: TaskState) -> TeamPreferenceQueryParameters:
        non_empty_teams = [
            name for name, members in self.team_assignment.items() if members
        ]
        if not non_empty_teams:
            raise RuntimeError("The team assignment is empty")
        team_name = random.choice(non_empty_teams)
        members = self.team_assignment[team_name]
        preferences = state.relations.get("coffee_preference", {})
        known = {
            name: preferences[name]
            for name in members
            if name in preferences
        }
        focus = random.choice(self._get_available_focus(state, members))
        instruction = AskTeamCoffeePreferencesStage._build_question(
            team_name, focus
        )
        answer = AskTeamCoffeePreferencesStage._build_answer(
            team_name, members, known, focus
        )
        return TeamPreferenceQueryParameters(
            team_name,
            tuple(members),
            tuple(known.items()),
            focus,
            instruction,
            answer,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: TeamPreferenceQueryParameters,
    ) -> List[BaseTaskStage]:
        return [
            AskTeamCoffeePreferencesStage(
                parameters.team_name,
                list(parameters.members),
                dict(parameters.known_preferences),
                parameters.focus,
            )
        ]

    def apply_request(
        self,
        state: TaskState,
        parameters: TeamPreferenceQueryParameters,
    ) -> TaskState:
        if state.properties.get(
            "team_coffee_preference_needs_application", False
        ):
            state.properties["team_coffee_preference_needs_application"] = False
            state.properties["team_coffee_preference_applications"] = (
                state.properties.get("team_coffee_preference_applications", 0)
                + 1
            )
        return state

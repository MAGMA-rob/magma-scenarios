from typing import Dict, List, Literal
import random
from collections import defaultdict

from magma_core.base.stage import BaseTaskStage
from magma_core.base.user_request import BaseRequest, BaseConstraintRequest
from magma_core.base.state import TaskState
from magma_core.base.data_structures import UserInstruction

from .coffee_stages import (
    MakeOneCoffeStage,
    CoffeeCompositeStage,
    AskPeopleInTeamStage,
    AskTeamsForPeopleStage,
    AskTeamCoffeePreferencesStage,
)

from magma_scenarios.templates.stages import MissingInformationStage

from .coffee_constraints import (
    CoffeePreferenceConstraint,
    TeamCoffeePreferenceConstraint,
)

class AskCoffeeRequest(BaseRequest):

    def __init__(self, max_coffee_making : int = 2, strict_order : bool = True):
        super().__init__()
        self.max_nb = max_coffee_making
        self.strict_order = strict_order
        if strict_order == False:
            raise ValueError("Strict order to True is not currently supported")

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:

        pods = state.attributes.get("coffee_pod",[])
        n = random.randint(1,self.max_nb)

        if len(pods) == 0:
            raise RuntimeError("Failed to find pods")

        pods_seq = random.choices(pods,k=n)
        stages = []
        coffee_str = ' and '.join(pods_seq)
        instruction = f"Hello, please make thse coffee in this exact order: {coffee_str}"
        for i in range(n):
            stages.append(MakeOneCoffeStage(
                pods_seq[i],
                instruction,
                [],
                flag_answer=i==n-1
            ))
            instruction="none"

        return stages

class GiveCoffeePreference(BaseConstraintRequest):

    def __init__(self, possible_names : List[str], max_name : int = 4, max_different_name : int = 6):
        super().__init__()
        self.max_nb = max_name
        self.possible_names = possible_names
        self.max_difference = max_different_name

    def initialize_constraints(self, state: TaskState):
        self.constraints = []

        coffee = state.attributes.get("coffee_pod",[])
        if len(coffee) <= 0:
            raise RuntimeError("Empty coffee pod")
        
        # If we are at the maximum of different preference we just modify existing ones
        if len(state.relations.get("coffee_preference",{})) >= self.max_difference:
            n = random.randint(1,min(len(self.possible_names),self.max_nb))
            names = random.sample(list(state.relations.get("coffee_preference",{}).keys()),k=n)
        else:
            # Otherwise we sample new ones
            max_diff = self.max_difference - len(state.relations.get("coffee_preference",{}))
            n = random.randint(1,min(len(self.possible_names),self.max_nb,max_diff))
            names = random.sample(self.possible_names,k=n)

        coffees = random.choices(coffee,k=n)
        
        assignment = defaultdict(list)
        for name, pod in zip(names,coffees):
            assignment[pod].append(name)
            self.constraints.append(
                CoffeePreferenceConstraint(
                    name,pod
                )
            )
        self.constraint_msg = "Hey, please remember that "
        for i, (pod, name_list) in enumerate(assignment.items()):
            names_str = " and ".join(name_list)
            self.constraint_msg += names_str + " like their coffee " + str(pod)
            if i != len(assignment)-1:
                self.constraint_msg += ", "
        self.constraint_msg += "."


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
        if self.mode == "override":
            return [
                team_name
                for team_name, members in self.team_assignment.items()
                if len(members) > 0
            ]

        preferences = state.relations.get("coffee_preference", {})
        return [
            team_name
            for team_name, members in self.team_assignment.items()
            if len(members) > 0 and any(member not in preferences for member in members)
        ]

    def sampling_weight(self, state: TaskState) -> float:
        if len(state.attributes.get("coffee_pod", [])) == 0:
            return 0
        return 1 if len(self._get_eligible_teams(state)) > 0 else 0

    def initialize_constraints(self, state: TaskState):
        self.constraints = []

        eligible_teams = self._get_eligible_teams(state)
        if len(eligible_teams) == 0:
            raise RuntimeError(f"Failed to sample a team in {self.__class__.__name__}")

        coffees = state.attributes.get("coffee_pod", [])
        if len(coffees) == 0:
            raise RuntimeError("Empty coffee pod")

        nb_team = random.randint(1, min(self.max_team_assignment, len(eligible_teams)))
        selected_teams = random.sample(eligible_teams, k=nb_team)
        selected_coffees = random.choices(coffees, k=nb_team)

        msg_parts = []
        for team_name, coffee in zip(selected_teams, selected_coffees):
            self.constraints.append(
                TeamCoffeePreferenceConstraint(
                    team_name=team_name,
                    team_members=self.team_assignment[team_name],
                    coffee=coffee,
                    mode=self.mode,
                )
            )
            if self.mode == "override":
                msg_parts.append(f"everyone in team {team_name} likes {coffee} coffee")
            else:
                msg_parts.append(
                    f"by default, members of team {team_name} like {coffee} coffee if they do not already have a known preference"
                )

        self.constraint_msg = "Hello, please remember that " + ", and ".join(msg_parts) + "."

class AskCoffeePerUser(BaseRequest):

    def __init__(
            self,
            possible_names : List[str],
            max_coffee : int = 3,
            force_order : bool = True
        ) -> None:
        super().__init__()
        self.nb = max_coffee
        self.possible_names = possible_names
        self.force_order = force_order

    def _join_names(self, names: List[str]) -> str:
        if len(names) == 1:
            return names[0]
        return " and ".join(names)

    def _build_missing_preference_answer(self, missing_names: List[str]) -> str:
        joined_names = self._join_names(missing_names)
        if len(missing_names) == 1:
            return f"The model must inform that {joined_names} does not have any coffee preference"
        return f"The model must inform that {joined_names} do not have any coffee preference"

    def _build_preference_resolution(self, missing_assignment: Dict[str, str]) -> str:
        if len(missing_assignment) == 0:
            return ""

        parts = [f"{name} wants a {pod} coffee" for name, pod in missing_assignment.items()]
        if len(parts) == 1:
            return parts[0] + "."
        if len(parts) == 2:
            return " and ".join(parts) + "."
        return ", ".join(parts[:-1]) + f", and {parts[-1]}."

    def _build_ordered_instruction(self, names: List[str]) -> str:
        if len(names) == 1:
            return f"Please make the coffee for {names[0]}."

        parts = [f"first the coffee for {names[0]}"]
        for name in names[1:]:
            parts.append(f"then the one for {name}")

        if len(parts) == 2:
            ordered_names = " and ".join(parts)
        else:
            ordered_names = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        return f"Please make {ordered_names}."

    def _sample_requested_users(self, state: TaskState) -> tuple[List[str], Dict[str, str]]:
        pods = state.attributes.get("coffee_pod", [])
        if len(pods) == 0:
            raise RuntimeError("Empty coffee pod")

        preference: Dict[str, str] = state.relations.get("coffee_preference", {})
        known_names = list(preference.keys())
        random.shuffle(known_names)

        all_names = known_names.copy()
        for name in self.possible_names:
            if name not in preference and name not in all_names:
                all_names.append(name)

        if len(all_names) == 0:
            raise RuntimeError("Failed to sample a user to serve coffee")

        n = random.randint(1, min(self.nb, len(all_names)))

        selected_names = known_names[:min(n, len(known_names))]
        if len(selected_names) < n:
            missing_names = [name for name in all_names if name not in preference]
            random.shuffle(missing_names)
            selected_names.extend(missing_names[:n - len(selected_names)])

        missing_assignment = {
            name: random.choice(pods)
            for name in selected_names
            if name not in preference
        }
        return selected_names, missing_assignment

    def _get_capsule(
            self,
            name: str,
            preference: Dict[str, str],
            missing_assignment: Dict[str, str]
        ) -> str:
        if name in missing_assignment:
            return missing_assignment[name]
        return preference[name]

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        stages = []

        preference: Dict[str, str] = state.relations.get("coffee_preference", {})
        names, missing_assignment = self._sample_requested_users(state)

        request_instruction = f"Please make coffee for {self._join_names(names)}."
        if missing_assignment:
            stages.append(MissingInformationStage(
                UserInstruction(request_instruction),
                self._build_missing_preference_answer(list(missing_assignment.keys())),
                state.memory,
                state.attributes
            ))

        ordered_instruction = self._build_ordered_instruction(names)
        resolution_instruction = self._build_preference_resolution(missing_assignment)

        if self.force_order:
            first_instruction = ordered_instruction
            if resolution_instruction:
                first_instruction = f"{resolution_instruction} {ordered_instruction}"

            for i, name in enumerate(names):
                capsule = self._get_capsule(name, preference, missing_assignment)
                stages.append(MakeOneCoffeStage(
                    capsule,
                    first_instruction if i == 0 else "none",
                    [],
                    i == len(names) - 1
                ))
            return stages

        if len(names) == 1:
            name = names[0]
            capsule = self._get_capsule(name, preference, missing_assignment)
            instruction = resolution_instruction if resolution_instruction else request_instruction
            stages.append(MakeOneCoffeStage(capsule, instruction, [], True))
            return stages

        number = defaultdict(int)
        for name in names:
            number[self._get_capsule(name, preference, missing_assignment)] += 1
        stages.append(CoffeeCompositeStage(dict(number)))

        return stages
    
class AskPeopleTeam(BaseRequest):

    def __init__(self, team_assignment : Dict[str, List[str]], max_nb : int = 2) -> None:
        super().__init__()
        self.max_nb = max_nb
        self.team_assignment = team_assignment
        self.people_to_team: Dict[str, str] = {}

        for team_name, members in team_assignment.items():
            for member in members:
                if member in self.people_to_team:
                    raise ValueError(f"{member} is assigned to multiple teams")
                self.people_to_team[member] = team_name

    def sampling_weight(self, state: TaskState) -> float:
        return 1 if len(self.people_to_team) > 0 else 0

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        if len(self.people_to_team) == 0:
            raise RuntimeError("Failed to sample a person because the team assignment is empty")

        nb = random.randint(1, min(self.max_nb, len(self.people_to_team)))
        names = random.sample(list(self.people_to_team.keys()), nb)
        teams = [self.people_to_team[name] for name in names]

        return [AskTeamsForPeopleStage(names, teams)]


class AskPeopleInTeam(BaseRequest):

    def __init__(self, team_assignment : Dict[str, List[str]]) -> None:
        super().__init__()
        self.team_assignment = team_assignment

    def sampling_weight(self, state: TaskState) -> float:
        return 1 if any(len(members) > 0 for members in self.team_assignment.values()) else 0

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        non_empty_teams = [
            team_name
            for team_name, members in self.team_assignment.items()
            if len(members) > 0
        ]
        if len(non_empty_teams) == 0:
            raise RuntimeError("Failed to sample a team because the team assignment is empty")

        team_name = random.choice(non_empty_teams)
        members = self.team_assignment[team_name]

        return [AskPeopleInTeamStage(team_name, members)]


class AskCoffeePreferenceInTeam(BaseRequest):

    def __init__(
            self,
            team_assignment: Dict[str, List[str]],
            allowed_focus: List[str] | None = None,
        ) -> None:
        super().__init__()
        self.team_assignment = team_assignment
        self.allowed_focus = allowed_focus

    def _get_available_focus(
            self,
            state: TaskState,
            team_members: List[str],
        ) -> List[str]:
        preferences: Dict[str, str] = state.relations.get("coffee_preference", {})
        coffee_pods = state.attributes.get("coffee_pod", [])

        if self.allowed_focus is None:
            candidate_focus = ["all", *coffee_pods, "unknown"]
        else:
            candidate_focus = self.allowed_focus

        available_focus = []
        for focus in candidate_focus:
            if focus == "all":
                available_focus.append(focus)
            elif focus == "unknown":
                if any(name not in preferences for name in team_members):
                    available_focus.append(focus)
            elif any(preferences.get(name) == focus for name in team_members):
                available_focus.append(focus)

        if len(available_focus) == 0:
            return ["all"]
        return available_focus

    def sampling_weight(self, state: TaskState) -> float:
        return 1 if any(len(members) > 0 for members in self.team_assignment.values()) else 0

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        non_empty_teams = [
            team_name
            for team_name, members in self.team_assignment.items()
            if len(members) > 0
        ]
        if len(non_empty_teams) == 0:
            raise RuntimeError("Failed to sample a team because the team assignment is empty")

        team_name = random.choice(non_empty_teams)
        team_members = self.team_assignment[team_name]
        preferences = state.relations.get("coffee_preference", {})
        known_preferences = {
            name: preferences[name]
            for name in team_members
            if name in preferences
        }

        focus = random.choice(self._get_available_focus(state, team_members))

        return [
            AskTeamCoffeePreferencesStage(
                requested_team=team_name,
                team_members=team_members,
                known_preferences=known_preferences,
                focus=focus,
            )
        ]

from magma_core.simulation.stage import (
    BaseTaskStage, AskingBaseStage,
    StageErrorParameters, StageGlobalParameters
)
from magma_core.simulation.utils.env_utils import is_object_inside_target
from magma_core.utils.text_utils import join_with_and, is_or_are
from magma_core.simulation.data_structures import (
    UserInstruction, EmptyInstruction,
    Log, StageInput
)
from magma_core.simulation.goals import BaseGoal

from magma_scenarios.utils import sapien_to_tensor
from magma_scenarios.templates.errors import OneShotToolFailureError


from .attributes import att, dropped_mug_pose, loaded_capsule_pose

import torch
from typing import List, Dict, Optional

class MugAndCapsuleGoal(BaseGoal):
    """Specific coffee scenario goal. Adding a relative add to the position for the target"""

    def __init__(self, capsule : str) -> None:
        super().__init__("Mug And Capsule positioned",None)
        self.capsule_target = sapien_to_tensor(loaded_capsule_pose)
        self.mug_target = sapien_to_tensor(dropped_mug_pose)
        self.capsule = capsule

    def verify(self, obs: Dict) -> torch.Tensor:
        device = obs["extra"]["coffee_maker"].device

        # check if the waited capsule is placed
        is_capsule_loaded = is_object_inside_target(
            obs["extra"][self.capsule], 
            torch.add(obs["extra"]["coffee_maker"][:,:7], self.capsule_target.to(device)),
            0.06)
        
        # check if the mug is placed
        is_mug_dropped = is_object_inside_target(
            obs["extra"]["mug"], 
            torch.add(obs["extra"]["coffee_maker"][:,:7], self.mug_target.to(device)),
            0.06)
        
        return (is_capsule_loaded & is_mug_dropped).int()

    def _to_spec_arguments(self) -> Dict:
        return {"capsule": self.capsule}

class RefuseCoffee(BaseTaskStage):

    target_tool_calls = 1
    max_tool_calls = 1

    def __init__(self, instruction: str, verif_prompt : str) -> None:
        super().__init__(
            [],
            "The robot must refuse to make a coffee for the user.",
            StageInput(UserInstruction(instruction),False),
            StageGlobalParameters(
                reset_at_end=False,
                verification_prompt= verif_prompt
            )
        )

    def _to_spec_arguments(self) -> Dict:
        return {
            "instruction": self.get_stage_input().instruction.get_content(),
            "verif_prompt": self.get_verification_prompt(),
        }

class MakeOneCoffeStage(BaseTaskStage):

    target_tool_calls = 3
    max_tool_calls = 4

    def __init__(
            self,
            capsule: str,
            instruction: str,
            flag_answer: bool,
            preliminary_tool_calls: int = 0,
            linked_to_prev: Optional[bool] = None,
        ) -> None:
        if not capsule in att["coffee_pod"]:
            raise ValueError(f"capsule {capsule} is not a known taste. Available are : {att['coffee_pod']}")
        if preliminary_tool_calls < 0:
            raise ValueError("preliminary_tool_calls must be non-negative")

        self.target_tool_calls = (
            3 + int(flag_answer) + preliminary_tool_calls
        )
        self.max_tool_calls = max(4, self.target_tool_calls)
        super().__init__(
            [],
            f"The goal of this stage is to add the mug to the machine, load a {capsule} pod and start the coffee machine.",
            StageInput(
                instruction=UserInstruction(instruction) if instruction != "none" else EmptyInstruction(),
                flag_answer_to_user=flag_answer,
                linked_to_prev=linked_to_prev,
            ),
            StageGlobalParameters(reset_at_end=True),
            error_parameters=StageErrorParameters(
                possible_errors=[OneShotToolFailureError()]
            ),
        )
        self.capsule = capsule
        self.preliminary_tool_calls = preliminary_tool_calls

    def _to_spec_arguments(self) -> Dict:
        instruction = self.get_stage_input().instruction
        return {
            "capsule": self.capsule,
            "instruction": (
                "none"
                if isinstance(instruction, EmptyInstruction)
                else instruction.get_content()
            ),
            "flag_answer": self.get_stage_input().flag_answer_to_user,
            "preliminary_tool_calls": self.preliminary_tool_calls,
            "linked_to_prev": self.get_stage_input().linked_to_prev,
        }

    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        """
        Check if the press button correctly happens after mug and pods placed.
        """
        loaded_pods = []
        is_mug_placed = False

        for l in stage_log:
            task_name = l.function
            if task_name == "load_capsule": loaded_pods.append(l.content)
            if task_name == "place_mug": is_mug_placed = True
            if task_name == "press_button":
                if is_mug_placed and loaded_pods == [self.capsule]:
                    return 1
                return -1
        return 0
    
class LookupTeamThenAnswerStage(BaseTaskStage):
    """Look up one person's team before giving a textual answer."""

    target_tool_calls = 2
    max_tool_calls = 2

    def __init__(
            self,
            instruction: str,
            expected_answer: str,
        ) -> None:
        super().__init__(
            goals=[],
            stage_goal_description=(
                "Find the person's team before answering: "
                f"{expected_answer}"
            ),
            stage_input=StageInput(
                instruction=UserInstruction(instruction),
                flag_answer_to_user=False,
            ),
            global_parameters=StageGlobalParameters(
                reset_at_end=True,
                verification_prompt=expected_answer,
                allow_tools_before_answer=True,
                allowed_tools=["team_from_people"],
            ),
        )
        self.instruction = instruction
        self.expected_answer = expected_answer

    def _to_spec_arguments(self) -> Dict:
        return {
            "instruction": self.instruction,
            "expected_answer": self.expected_answer,
        }


class AskPeopleInTeamStage(AskingBaseStage):

    target_tool_calls = 2
    max_tool_calls = 2

    def __init__(self, requested_team : str, member_list : List[str]) -> None: 
        if len(member_list) == 0:
            raise ValueError("member_list must not be empty")
        members = join_with_and(member_list)
        be_verb = is_or_are(member_list)
        super().__init__(
            question=f"Who is in team {requested_team}?",
            answer=f"{members} {be_verb} in team {requested_team}.",
            linked_to_prev=False,
            allow_tools_before_answer=True,
            allowed_tools=["people_from_team"]
        )
        self.stage_goal_description = f"The goal of the stage is to ensure that the model call the tool to fetch team information before answering that : {members} {be_verb} in team {requested_team}"
        self.requested_team = requested_team
        self.member_list = list(member_list)

    def _to_spec_arguments(self) -> Dict:
        return {
            "requested_team": self.requested_team,
            "member_list": list(self.member_list),
        }


class AskTeamsForPeopleStage(AskingBaseStage):

    def __init__(self, requested_people : List[str], teams_associated : List[str]) -> None:
        if len(requested_people) != len(teams_associated):
            raise ValueError(
                "requested_people and teams_associated must have the same length"
            )
        if len(requested_people) == 0:
            raise ValueError("requested_people must not be empty")

        if len(requested_people) == 1:
            question = f"What team does {requested_people[0]} belong to?"
        else:
            question = f"In which teams are {join_with_and(requested_people)}?"

        answer_parts = [
            f"{person} is in team {team}"
            for person, team in zip(requested_people, teams_associated)
        ]
        self.target_tool_calls = len(requested_people) + 1
        self.max_tool_calls = self.target_tool_calls
        super().__init__(
            question=question,
            answer=join_with_and(answer_parts) + ".",
            linked_to_prev=False,
            allow_tools_before_answer=True,
            allowed_tools=["team_from_people"]
        )
        self.stage_goal_description = f"The goal of the stage is to ensure that the model call the tool to fetch team information before answering that : {self.global_parameters.verification_prompt}"
        self.requested_people = list(requested_people)
        self.teams_associated = list(teams_associated)

    def _to_spec_arguments(self) -> Dict:
        return {
            "requested_people": list(self.requested_people),
            "teams_associated": list(self.teams_associated),
        }


class AskCoffeePreferenceForUserStage(AskingBaseStage):

    target_tool_calls = 1
    max_tool_calls = 1

    def __init__(
            self,
            requested_person: str,
            coffee_preference: Optional[str],
            question: Optional[str] = None,
        ) -> None:
        if requested_person == "":
            raise ValueError("requested_person must not be empty")

        if question is None:
            raise ValueError("question must be sampled before creating the stage")

        if coffee_preference is None:
            answer = f"{requested_person} does not have any known coffee preference."
        else:
            answer = f"{requested_person} likes {coffee_preference} coffee."

        super().__init__(
            question=question,
            answer=answer,
            linked_to_prev=False,
        )
        self.requested_person = requested_person
        self.coffee_preference = coffee_preference

    def _to_spec_arguments(self) -> Dict:
        return {
            "requested_person": self.requested_person,
            "coffee_preference": self.coffee_preference,
            "question": self.question,
        }


class AskTeamCoffeePreferencesStage(AskingBaseStage):

    target_tool_calls = 2
    max_tool_calls = 2

    def __init__(
            self,
            requested_team: str,
            team_members: List[str],
            known_preferences: Dict[str, str],
            focus: str = "all"
        ) -> None:
        if len(team_members) == 0:
            raise ValueError("team_members must not be empty")

        question = self._build_question(requested_team, focus)
        answer = self._build_answer(requested_team, team_members, known_preferences, focus)

        super().__init__(
            question=question,
            answer=answer,
            linked_to_prev=False,
            allow_tools_before_answer=True,
            allowed_tools=["people_from_team"],
        )
        self.stage_goal_description = f"The goal of the stage is to ensure that the model call the tool to fetch team information before answering that : {self.global_parameters.verification_prompt}"
        self.requested_team = requested_team
        self.team_members = list(team_members)
        self.known_preferences = dict(known_preferences)
        self.focus = focus

    def _to_spec_arguments(self) -> Dict:
        return {
            "requested_team": self.requested_team,
            "team_members": list(self.team_members),
            "known_preferences": dict(self.known_preferences),
            "focus": self.focus,
        }

    @staticmethod
    def _build_question(requested_team: str, focus: str) -> str:
        if focus == "all":
            return f"What are the coffee preferences of the members of team {requested_team}?"
        if focus == "unknown":
            return f"Which members of team {requested_team} do not have a known coffee preference?"
        return f"Which members of team {requested_team} like {focus} coffee?"

    @classmethod
    def _build_answer(
            cls,
            requested_team: str,
            team_members: List[str],
            known_preferences: Dict[str, str],
            focus: str,
        ) -> str:
        if focus == "all":
            return cls._build_full_preference_answer(requested_team, team_members, known_preferences)
        if focus == "unknown":
            return cls._build_unknown_preference_answer(requested_team, team_members, known_preferences)
        return cls._build_filtered_preference_answer(requested_team, team_members, known_preferences, focus)

    @staticmethod
    def _build_full_preference_answer(
            requested_team: str,
            team_members: List[str],
            known_preferences: Dict[str, str],
        ) -> str:
        parts = [
            f"{name} likes {known_preferences[name]} coffee"
            for name in team_members
            if name in known_preferences
        ]
        unknown_people = [name for name in team_members if name not in known_preferences]

        if len(unknown_people) == len(team_members):
            return f"No member of team {requested_team} has a known coffee preference."

        if len(unknown_people) == 1:
            parts.append(f"{unknown_people[0]} does not have any known coffee preference")
        elif len(unknown_people) > 1:
            parts.append(
                f"{join_with_and(unknown_people)} do not have any known coffee preference"
            )

        return f"In team {requested_team}, {join_with_and(parts)}."

    @staticmethod
    def _build_unknown_preference_answer(
            requested_team: str,
            team_members: List[str],
            known_preferences: Dict[str, str],
        ) -> str:
        unknown_people = [name for name in team_members if name not in known_preferences]
        if len(unknown_people) == 0:
            return f"All members of team {requested_team} have a known coffee preference."
        if len(unknown_people) == 1:
            return f"{unknown_people[0]} does not have any known coffee preference in team {requested_team}."
        return (
            f"{join_with_and(unknown_people)} do not have any known coffee preference "
            f"in team {requested_team}."
        )

    @staticmethod
    def _build_filtered_preference_answer(
            requested_team: str,
            team_members: List[str],
            known_preferences: Dict[str, str],
            focus: str,
        ) -> str:
        matching_people = [
            name
            for name in team_members
            if known_preferences.get(name) == focus
        ]
        if len(matching_people) == 0:
            return f"No member of team {requested_team} has a known preference for {focus} coffee."
        if len(matching_people) == 1:
            return f"{matching_people[0]} likes {focus} coffee in team {requested_team}."
        return f"{join_with_and(matching_people)} like {focus} coffee in team {requested_team}."

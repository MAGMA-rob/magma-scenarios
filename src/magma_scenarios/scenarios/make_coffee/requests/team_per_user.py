from dataclasses import dataclass
import random
from typing import Dict, List

from magma_core.simulation.data_structures import UserInstruction
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_core.simulation.requests import BaseRequest
from magma_scenarios.templates.stages import ForbiddenElemStage

from ..coffee_constraints import get_unavailable_coffee_pods
from ..coffee_stages import LookupTeamThenAnswerStage, MakeOneCoffeStage
from .common import (
    missing_preference_verification,
    preference_resolution,
    unavailable_people_verification,
)


@dataclass(frozen=True)
class TeamCoffeeParameters:
    name: str
    team_name: str
    coffee: str
    instruction: str
    next_instruction: str
    has_team_rule: bool
    unavailable: bool


class AskTeamCoffeePerUser(BaseRequest[TeamCoffeeParameters]):
    """Make coffee after explicitly resolving the person's team."""

    def __init__(self, team_assignment: Dict[str, List[str]]) -> None:
        super().__init__()
        self.people_to_team: Dict[str, str] = {}
        for team_name, members in team_assignment.items():
            for member in members:
                if member in self.people_to_team:
                    raise ValueError(f"{member} is assigned to multiple teams")
                self.people_to_team[member] = team_name

    def sampling_weight(self, state: TaskState) -> float:
        if not self.people_to_team or not state.attributes.get("coffee_pod", []):
            return 0
        if state.properties.get(
            "team_coffee_preference_needs_application", False
        ):
            return 6
        return 2

    def sample_parameters(self, state: TaskState) -> TeamCoffeeParameters:
        if not self.people_to_team:
            raise RuntimeError("The team registry is empty")
        name = random.choice(list(self.people_to_team))
        team_name = self.people_to_team[name]
        instruction = f"Please make coffee for {name}."
        team_rule = state.relations.get(
            "team_coffee_preference_rules", {}
        ).get(team_name)
        coffee = (
            random.choice(state.attributes["coffee_pod"])
            if team_rule is None
            else team_rule["coffee"]
        )
        return TeamCoffeeParameters(
            name,
            team_name,
            coffee,
            instruction,
            (
                preference_resolution({name: coffee})
                if team_rule is None
                else instruction
            ),
            team_rule is not None,
            coffee in set(get_unavailable_coffee_pods(state)),
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: TeamCoffeeParameters,
    ) -> List[BaseTaskStage]:
        stages: List[BaseTaskStage] = []
        if not parameters.has_team_rule:
            stages.append(
                LookupTeamThenAnswerStage(
                    parameters.instruction,
                    missing_preference_verification([parameters.name]),
                )
            )
        if parameters.unavailable:
            if parameters.has_team_rule:
                stages.append(
                    LookupTeamThenAnswerStage(
                        parameters.instruction,
                        unavailable_people_verification(
                            [parameters.name],
                            {parameters.name: parameters.coffee},
                        ),
                    )
                )
            else:
                stage = ForbiddenElemStage(
                    UserInstruction(
                        parameters.next_instruction,
                        has_constraint=True,
                    ),
                    unavailable_people_verification(
                        [parameters.name],
                        {parameters.name: parameters.coffee},
                    ),
                )
                stage.stage_input.linked_to_prev = True
                stages.append(stage)
            return stages

        make_stage = MakeOneCoffeStage(
            parameters.coffee,
            parameters.next_instruction,
            flag_answer=True,
            preliminary_tool_calls=1 if parameters.has_team_rule else 0,
        )
        if stages:
            make_stage.stage_input.linked_to_prev = True
        stages.append(make_stage)
        return stages

    def apply_request(
        self,
        state: TaskState,
        parameters: TeamCoffeeParameters,
    ) -> TaskState:
        if state.properties.get(
            "team_coffee_preference_needs_application", False
        ):
            state.properties["team_coffee_preference_needs_application"] = False
        return state

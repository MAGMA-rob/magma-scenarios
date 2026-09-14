# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
# Arthur TANNEAU

from pathlib import Path

from magma_core.simulation.tasks import TaskDefinition
from magma_core.simulation.state import TaskState
from magma_core.simulation.data_structures import SituationInit

from .simple_tool import MakingCoffeeTool

from .attributes import (
    att, loaded_capsule_pose, dropped_mug_pose,
    people, build_people_assignment, sapien
)
from .coffee_request import (
    GiveCoffeePreference,
    GiveTeamCoffeePreference,
    AskCoffeeRequest,
    AskCoffeePerUser,
    AskTeamCoffeePerUser,
    AskPeopleInTeam,
    AskPeopleTeam,
    AskCoffeePreferenceForUser,
    AskCoffeePreferenceInTeam, #some bugs here
    ToggleCoffeeAvailability,
    CoffeeInterruptionRequest,
)
from .rule_renderer import CoffeeRuleRenderer

import random

RANDOMIZED_CONFIG_PATH = str(Path(__file__).resolve().parent / "make_coffee_cfg.yaml")

class SimpleDefinition(TaskDefinition):
    maniskill_env_id = "MakeCoffee-v1"
    randomized_config_path = RANDOMIZED_CONFIG_PATH
    Tools_cls = MakingCoffeeTool
    RuleRenderer_cls = CoffeeRuleRenderer

    active_requests = [
        GiveCoffeePreference(people),
        ToggleCoffeeAvailability(),
        AskCoffeePreferenceForUser(people),
        AskCoffeeRequest(),
        AskCoffeePerUser(people, force_order=True),
        CoffeeInterruptionRequest(people,max_coffee=5),
    ]

    def __init__(self) -> None:  
        super().__init__(
            name="Simple definition",
            situation_init=SituationInit(
                att,
                memory={"memory_list":[
                    (
                        "To prepare a coffee, place the mug, load the appropriate "
                        "capsule, then start the coffee maker."
                    ),
                    (
                        "When asked to make coffee for a person whose preference "
                        "is unknown, ask the user for that person's coffee "
                        "preference."
                    ),
                ]}
            ),
            randomized_config_path=RANDOMIZED_CONFIG_PATH,
            tools_constant= {
                "loaded_capsule_pose": loaded_capsule_pose,
                "dropped_mug_pose" : dropped_mug_pose,
                "base_pose" : sapien.Pose(p = [-0.1,0,0.4],q = [0,1,0,0]),
                "teams" : {}},
        )

        self.starting_state = TaskState()
        self.starting_state.attributes = att
        self.starting_state.relations = {
            "coffee_preference": {},
            "team_coffee_preference_rules": {},
        }
        self.starting_state.properties.update({
            "unavailable_coffee_pods": [],
        })


class TeamDefinition(TaskDefinition):
    maniskill_env_id = "MakeCoffee-v1"
    randomized_config_path = RANDOMIZED_CONFIG_PATH
    Tools_cls = MakingCoffeeTool
    RuleRenderer_cls = CoffeeRuleRenderer

    def __init__(self)-> None:

        nb_team = random.randint(2,4)
        nb_people_per_team = random.randint(2,5)

        team_dict = build_people_assignment(nb_team,nb_people_per_team)

        super().__init__(
            name = "Team Definition",
            situation_init=SituationInit(
                att,
                memory={"memory_list":[
                    (
                        "To prepare a coffee, place the mug, load the appropriate "
                        "capsule, then start the coffee maker."
                    ),
                    (
                        "When asked to make coffee for a person, first use the "
                        "team registry to find their team, then apply that team's "
                        "coffee preference. If the team has no coffee preference, "
                        "ask the user for the person's preference."
                    ),
                ]}
            ),
            tools_constant={
                "loaded_capsule_pose": loaded_capsule_pose,
                "dropped_mug_pose" : dropped_mug_pose,
                "base_pose" : sapien.Pose(p = [-0.1,0,0.4],q = [0,1,0,0]),
                "teams" : team_dict},
        )

        self.starting_state = TaskState()
        self.starting_state.attributes = att
        self.starting_state.relations = {
            "coffee_preference": {},
            "team_coffee_preference_rules": {},
        }
        self.starting_state.properties.update({
            "unavailable_coffee_pods": [],
        })

        self.active_requests = [
            GiveTeamCoffeePreference(team_dict, mode="override"),
            GiveTeamCoffeePreference(team_dict, mode="default"),
            ToggleCoffeeAvailability(),
            AskTeamCoffeePerUser(team_dict),
            AskPeopleTeam(team_dict,3),
            AskPeopleInTeam(team_dict)
        ]

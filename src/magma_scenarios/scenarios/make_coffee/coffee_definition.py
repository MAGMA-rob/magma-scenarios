# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
# Arthur TANNEAU

from pathlib import Path

from magma_core.base.tasks import TaskDefinition
from magma_core.base.state import TaskState


from .simple_tool import MakingCoffeeTool
from .attributes import att, loaded_capsule_pose, dropped_mug_pose, people, build_people_assignment
from .coffee_request import (
    GiveCoffeePreference,
    GiveTeamCoffeePreference,
    AskCoffeeRequest,
    AskCoffeePerUser,
    AskPeopleInTeam,
    AskPeopleTeam,
    AskCoffeePreferenceInTeam,
    ToggleCoffeeAvailability,
)

import sapien
import random

RANDOMIZED_CONFIG_PATH = str(Path(__file__).resolve().parent / "make_coffee_cfg.yaml")

class SimpleDefinition(TaskDefinition):
    env_id = "MakeCoffee-v1"
    randomized_config_path = RANDOMIZED_CONFIG_PATH
    Tools_cls = MakingCoffeeTool

    active_requests = [
        GiveCoffeePreference(people),
        ToggleCoffeeAvailability(),
        AskCoffeeRequest(),
        AskCoffeePerUser(people, force_order=True)
    ]

    def __init__(self) -> None:  
        super().__init__(
            name="Simple definition",
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
    env_id = "MakeCoffee-v1"
    randomized_config_path = RANDOMIZED_CONFIG_PATH
    Tools_cls = MakingCoffeeTool

    def __init__(self)-> None:

        nb_team = random.randint(2,4)
        nb_people_per_team = random.randint(2,5)

        team_dict = build_people_assignment(nb_team,nb_people_per_team)
        all_people = []
        for k, v in team_dict.items():
            all_people.extend(v)

        super().__init__(
            name = "Team Definition",
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
            GiveCoffeePreference(all_people),
            GiveTeamCoffeePreference(team_dict, mode="override"),
            GiveTeamCoffeePreference(team_dict, mode="default"),
            ToggleCoffeeAvailability(),
            AskCoffeePerUser(all_people, force_order=True),
            AskPeopleTeam(team_dict,3),
            AskPeopleInTeam(team_dict),
            AskCoffeePreferenceInTeam(team_dict),
        ]

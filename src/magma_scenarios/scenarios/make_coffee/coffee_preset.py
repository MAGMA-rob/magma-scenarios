# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
# Arthur TANNEAU

from pathlib import Path

from magma_core.base.tasks import BaseTask
from magma_core.base.tasks_style import TaskStyle
from magma_core.base.data_structures import UserInstruction, EmptyInstruction, Log, Situation

from .simple_tool import MakingCoffeeTool
from .attributes import att, loaded_capsule_pose, dropped_mug_pose, people, teams
from .coffee_stages import MakeOneCoffeStage, ConstraintCoffeeStage, RefuseCoffee, CoffeeCompositeStage , AskTeamStage, AskPersonStage

import sapien, random
from typing import List, Dict, Any
from collections import defaultdict

RANDOMIZED_CONFIG_PATH = str(Path(__file__).resolve().parent / "make_coffee_cfg.yaml")


named_instructions = [
    "{name}: Can you make me a coffee?",
    "Hello, make me a coffee for {name}",
    "A coffee for {name} please",
    "{name}: I am tired, I need a coffee",
    "{name} want a coffee."
]

    
class BaseCoffee(BaseTask):
    env_id = "MakeCoffee-v1"

    randomized_config_path = RANDOMIZED_CONFIG_PATH

    tools_constant = {"loaded_capsule_pose": loaded_capsule_pose, "dropped_mug_pose" : dropped_mug_pose, "base_pose" : sapien.Pose(p = [-0.1,0,0.4],q = [0,1,0,0])}

    Tools_cls = MakingCoffeeTool
    all_task_attributes = att

class TestComposite(BaseCoffee):

    name = "Test composite"

    styles = []
    approximal_difficulty = "Medium"

    def __init__(self) -> None:
        super().__init__()
        self.stages = [
            MakeOneCoffeStage("milky","blabla",[],True),
            CoffeeCompositeStage({"milky":1,"black":1})
        ]
   
class ConstrainedPreset(BaseCoffee):
    """
    Task to make coffee under constraints.
    DIfficulty Medium to Hard.
    """

    styles = [
        TaskStyle.CONSTRAINED,
        TaskStyle.LONG_STAGE
    ]

    name = "Make Coffee under constraint"

    randomized_config_path = "" # we let like this because we hardcode coffee in arguments

    def __init__(
            self,
            instruction_which_must_fail : str = "Hello! Can you make me a black coffee please",
            instruction_which_must_succeed : str = "Oh, okay let's do a milky instead",
            coffee_to_make : str = "milky",
            constraints : List[str] = ["There is no black coffee anymore."]):
        """
        You can set one instruction for the model which need to fail (due to constraints) and set some constraints as user-taste preference or some taste shortage.
        You need also to define an instruction associated with the coffee_to_make which can be completed according to the constraint.
        Availaible coffee taste are : black, milky and white.

        Medium : 1 constraint, Hard : 2+
        """
        super().__init__()
        self.stages= [ConstraintCoffeeStage("To prepare a coffee you must have placed the mug and loaded the capsule before pressing the start button.")]
        for c in constraints:
            self.stages.append(ConstraintCoffeeStage(c))
        self.stages.append(RefuseCoffee(instruction_which_must_fail, verif_prompt="The robot must refuse to make a coffee for the user."))
        self.stages.append(MakeOneCoffeStage(coffee_to_make, instruction_which_must_succeed, [],flag_answer=True))

        if len(constraints) > 1:
            self.approximal_difficulty = "Hard"
        else:
            self.approximal_difficulty = "Medium"


class TeamCoffePreset(BaseCoffee):
    name = "team assinement inside coffe scenario"
    styles = []

    def __init__(self, nb_team : int = 2, nb_people_per_team : int = 4) :
        super().__init__()

        people_coppy = people.copy()
        teams_coppy = teams.copy()

        random.shuffle(people_coppy)
        random.shuffle(teams_coppy)

        if nb_team > len(teams_coppy):
            raise TypeError(f"Only {len(teams_coppy)} exists but you asked for {nb_team}")
        if nb_people_per_team * nb_team > len(people_coppy):
            raise TypeError(f"You asked for {nb_people_per_team} for {nb_team} but only {len(people_coppy)} \
                            people names exists ({nb_people_per_team*nb_team})")

        selected_people = people_coppy[:nb_people_per_team*nb_team]
        teams_selected = teams_coppy[:nb_team]

        team_dict = {}

        for i,team in enumerate(teams_selected) :
            start = i*nb_people_per_team
            end = (i+1)*nb_people_per_team
            team_dict[team] = selected_people[start:end]

        random_team = random.choice(teams_selected)
        assigned_members = team_dict[random_team]

        random_person = random.choice(selected_people)
        assigned_team = next((k for k,v in team_dict.items() if random_person in v),None)

        preference_coffee = {}
        coffee_pref = defaultdict(list)
        for members in team_dict.values():
            for person in members:
                c = random.choice(att["coffee_pod"])
                preference_coffee[person] = c
                coffee_pref[c].append(person)

        self.tools_constant["teams"] = team_dict
        coffee_sentence = "Hello, please remember that "
        for i, (c, names) in enumerate(coffee_pref.items()):
            coffee_sentence += ", ".join(names) + f" like their coffee {c}"
            if i < len(coffee_pref)-1:
                coffee_sentence += " and "
            else:
                coffee_sentence += "."

        self.stages = [ConstraintCoffeeStage(coffee_sentence)]
        self.stages.append(AskTeamStage(random_team,assigned_members))
        if assigned_team is not None :
            self.stages.append(AskPersonStage(random_person,assigned_team))
        for team_member in team_dict[random_team] :
            self.stages.append(MakeOneCoffeStage(
                instruction = random.choice(named_instructions).format(name = team_member),
                capsule = preference_coffee[team_member],
                add_memory = [],
                flag_answer = True
            ))
        self.approximal_difficulty = "Hard"

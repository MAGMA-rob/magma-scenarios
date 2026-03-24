# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
# Arthur TANNEAU

from magma_core.base.tasks import BaseTask
from magma_core.base.tasks_style import TaskStyle
from magma_core.base.data_structures import UserInstruction, EmptyInstruction, Log, Situation

from .simple_tool import MakingCoffeeTool
from .attributes import att, loaded_capsule_pose, dropped_mug_pose
from .coffee_stages import MakeOneCoffeStage, ConstraintCoffeeStage, RefuseCoffee, CoffeeCompositeStage

import sapien, torch, random
from typing import List, Dict, Any
from importlib import resources


named_instructions = [
    "{name}: Can you make me a coffee?",
    "Hello, make me a coffee for {name}",
    "A coffee for {name} please",
    "{name}: I am tired, I need a coffee",
    "{name} want a coffee."
]

    
class BaseCoffee(BaseTask):
    env_id = "MakeCoffee-v1"

    randomized_config_path = str(resources.files(__package__).joinpath("make_coffee_cfg.yaml"))

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


class MultipleUserPreset(BaseCoffee):
    """
    This task allow to define multiple user and their preference on coffee. 
    Then it will generate random instruction using the names you provide.

    Difficulty Hard
    """

    styles = [
        TaskStyle.LONG_STAGE,
        TaskStyle.CONSTRAINED
    ]

    name = "Multiple user ask for coffee"

    def __init__(self, nb_of_coffee : int = 2, names_preference : Dict = {"Diana":"black","Frederic":"milky"}) -> None:
        """
        You can initialize diverse names and their preference in the names_preference dict. You can put a name as key and a coffee preference as value.
        You can also set the nb_of_coffee that will be asked to complete the task.
        The task is considered as hard for nb_of_coffee >= 2.
        """
        super().__init__()
        s = "Hey here are some preferences from me and my friends : "
        i = len(names_preference)
        if i == 0:
            raise ValueError("You need to pass at least one name and one preference")
        cpt = 1
        for name, coffee in names_preference.items():
            s += f"{name} like its coffee {coffee}"
            cpt+=1
            if cpt != i:
                s+= ", "
            else:
                s+="."
        self.stages = [ConstraintCoffeeStage(s)]
        people = list(names_preference.keys())
        for i in range(nb_of_coffee):
            self.stages.append(
                MakeOneCoffeStage(
                    instruction=random.choice(named_instructions).format(name=people[i]),
                    capsule=names_preference[people[i]],
                    add_memory=[],
                    flag_answer= i == nb_of_coffee-1
                )
            )

        self.approximal_difficulty = "Hard"

        
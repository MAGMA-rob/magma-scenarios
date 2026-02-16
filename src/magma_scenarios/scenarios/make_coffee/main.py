# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
# Arthur TANNEAU

from magma_core.base.tasks import BaseTask, BaseTaskStage, ConstraintBaseStage
from magma_core.base.tasks_style import TaskStyle
from magma_core.utils.env_utils import is_object_inside_target
from magma_core.base.data_structures import UserInstruction, EmptyInstruction, Log, Situation

from magma_scenarios.utils import sapien_to_tensor

from .simple_tool import MakingCoffeeTool

import sapien, torch, random
from typing import List, Dict, Any
from importlib import resources

att = {"coffee_pod": ["black", "milky", "white"]}
# mug and capsule positions relative to the coffee maker
dropped_mug_pose = sapien.Pose(p=[-0.22, 0, -0.1], q = [0,1,0,0])
loaded_capsule_pose = sapien.Pose(p=[-0.022, 0.04, 0.18], q = [0,1,0,0])

named_instructions = [
    "{name}: Can you make me a coffee?",
    "Hello, make me a coffee for {name}",
    "A coffee for {name} please",
    "{name}: I am tired, I need a coffee",
    "{name} want a coffee."
]

class ConstraintCoffeeStage(ConstraintBaseStage):

    def __init__(self, constraint: str) -> None:
        mem = [
            "You are in charge of making a coffee.",
            "To prepare a coffee you must have placed the mug and loaded the capsule before pressing the start button."
            ]
        super().__init__(constraint, mem, att)

class RefuseCoffee(BaseTaskStage):

    acceptance_steps = 0
    target_steps = 1

    def __init__(self, instruction, verif_prompt : str) -> None:
        super().__init__(False, "")

        self.situation = Situation(
            memory=[],
            preserved_memory_indices=[],
            attributes=att,
            flag_answer_to_user=False,
            instruction=UserInstruction(instruction),
        )

        self.verification_prompt = verif_prompt

class MakeOneCoffeStage(BaseTaskStage):

    target_steps = 3
    acceptance_steps = 1

    def __init__(self, capsule : str, instruction : str, add_memory : List[str]) -> None:
        super().__init__(True, f"The goal of this stage is to add the mug to the machine, load a {capsule} pod and start the coffee machine.")

        if not capsule in att["coffee_pod"]:
            raise ValueError(f"capsule {capsule} is not a known taste. Available are : {att['coffee_pod']}")
        self.capsule_to_load = capsule

        mem = [
            "You are in charge of making a coffee.",
            "To prepare a coffee you must have placed the mug and loaded the capsule before pressing the start button."
            ]
        mem.extend(add_memory)

        self.situation = Situation(
            memory=mem,
            instruction=UserInstruction(instruction) if instruction != "none" else EmptyInstruction(),
            attributes=att,
            flag_answer_to_user=False,
            preserved_memory_indices=[0,1]
        )

        self.capsule_target = sapien_to_tensor(loaded_capsule_pose)
        self.mug_target = sapien_to_tensor(dropped_mug_pose)

    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        """ Check if mug and capsule are correctly loaded"""

        device = obs["extra"]["coffee_maker"].device

        # check if the waited capsule is placed
        is_capsule_loaded = is_object_inside_target(
            obs["extra"][self.capsule_to_load], 
            torch.add(obs["extra"]["coffee_maker"][:,:7], self.capsule_target.to(device)),
            0.03)
        
        # check if the mug is placed
        is_mug_dropped = is_object_inside_target(
            obs["extra"]["mug"], 
            torch.add(obs["extra"]["coffee_maker"][:,:7], self.mug_target.to(device)),
            0.03)
        
        return (is_capsule_loaded & is_mug_dropped).int()
    
    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        """
        Check if the press button correctly happens after mug and pods placed.
        """        
        is_pods_load = False
        is_mug_placed = False

        for l in full_log:
            task_name = l.function
            if task_name == "load_capsule": is_pods_load = True
            if task_name == "place_mug": is_mug_placed = True
            if task_name == "press_button":
                if is_mug_placed and is_pods_load:
                    return 1
                return -1
        return 0
    
    def combine_stage_completion(self, task_completion: int, log_completion: int) -> int:
        """
        Here it's a special case, if the log return 1 and the env 0, it's a -1 (because maybe the pod or the mug was not correctly placed - the action failed and we press)
        """
        if log_completion == 1:
            if task_completion == 1:
                return 1
            return -1 # case where the env failed
        if task_completion == -1 or log_completion == -1:
            return -1
        return 0
    
class BaseCoffee(BaseTask):
    env_id = "MakeCoffee-v1"

    randomized_config_path = str(resources.files(__package__).joinpath("make_coffee_cfg.yaml"))

    tools_constant = {"loaded_capsule_pose": loaded_capsule_pose, "dropped_mug_pose" : dropped_mug_pose, "base_pose" : sapien.Pose(p = [-0.1,0,0.4],q = [0,1,0,0])}

    Tools_cls = MakingCoffeeTool
    all_task_attributes = att

class MakeCoffeeBasic(BaseCoffee):
    """
    Task description:
    The task is to make a coffee. The robot must choose and load one of the three available capsule,
    drop the mug in front of the coffee maker and press the coffee maker button to make a coffee.
    The mug and capsules are represented by coloured cube (blue for the mug and brown shades for
    the capsule) and the coffee maker is an articulated sapien object.

    Difficulty from Easy to Medium
    """

    name : str = "Making coffee"

    styles = [
        TaskStyle.MEMORY_COHERENCE,
        TaskStyle.LONG_STAGE
    ]

    def __init__(self, instruction : str = "Can you make me a black coffee please", capsule_list : List[str] = ["black"]):
        """
        You can set the instruction for the model and set accordingly a list of coffee you want. The instruction must ask for the same number of coffee than the length of the capsule_list.
        Availaible coffee taste are : black, milky and white.

        Easy : One coffee, Medium : 1-3 coffee, Hard : 4+
        """
        super().__init__()
        self.stages = [
            ConstraintCoffeeStage("To prepare a coffee you must have placed the mug and loaded the capsule before pressing the start button."),
            MakeOneCoffeStage(capsule_list[0], instruction, [])
        ]
        for i in range(1,len(capsule_list)):
            self.stages.append(MakeOneCoffeStage(capsule_list[i], "none", []))

        if len(capsule_list) > 3:
            self.approximal_difficulty = "Hard"
        elif len(capsule_list) > 1:
            self.approximal_difficulty = "Medium"
        else:
            self.approximal_difficulty = "Easy"
    
class ConstrainedCoffee(BaseCoffee):
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
        self.stages.append(MakeOneCoffeStage(coffee_to_make, instruction_which_must_succeed, []))

        if len(constraints) > 1:
            self.approximal_difficulty = "Hard"
        else:
            self.approximal_difficulty = "Medium"


class MultipleUserCoffee(BaseCoffee):
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
                    add_memory=[]
                )
            )

        self.approximal_difficulty = "Hard"

        
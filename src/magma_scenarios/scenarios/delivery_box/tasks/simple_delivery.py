# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import sapien, torch
from typing import Any, List, Dict, Tuple
from pathlib import Path
import random

from magma_core.base.tasks import BaseTask, BaseTaskStage
from magma_core.base.data_structures import UserInstruction, Log, TemplateInstruction, Situation, Instruction
from magma_core.base.tasks_style import TaskStyle
from magma_core.utils.env_utils import craft_random_manu_order

from ..tools.simple_tool import CycleTool

att = {"product_type":["coca","icetea","brets","donut"]}

class CycleStage(BaseTaskStage):
    """
   Task STage to launch a cycle according to a reference, a manufacturing order and a number of delivery.
    """

    target_steps = 1
    acceptance_steps = 1

    def __init__(self, recipe : List[str], manufacturing_order : str, deliveries : int, instruction : str) -> None:
        self.recipe = recipe
        self.manufacturing_order = manufacturing_order
        self.deliveries = deliveries

        self.situation = Situation(
            memory = ["You must make a delivery box by packing inside some objects."],
            preserved_memory_indices=[0],
            attributes=att,
            instruction=UserInstruction(instruction),
            flag_answer_to_user=True
        )

        super().__init__(True, f"The goal of the stage is to launch a cycle respecting recipe={self.recipe}, delivery_number={self.deliveries}, manufacturing_order={self.manufacturing_order}")


    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        return super().verif_env_completion(obs) # TO DO
    
    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if len(stage_log) == 0:
            return 0
        if stage_log[-1].content == (self.deliveries, self.manufacturing_order):
            return 1
        return -1
    
class ConstraintBaseStage(BaseTaskStage):
    """
    Text only stage for giving constraints
    """

    target_steps = 1
    acceptance_steps = 0

    def __init__(self, instruction : Instruction, reset_at_end: bool = False) -> None:
        if isinstance(instruction, TemplateInstruction):
            ver = f"The model must understand that it needs to update its recipe"
        else:
            ver = f"The model must aknowledge the constraint {instruction.get_content()}"
        super().__init__(reset_at_end, ver)

        self.situation = Situation(
            memory = ["You must make a delivery box by packing inside some objects."],
            preserved_memory_indices=[0],
            attributes=att,
            instruction=instruction,
            flag_answer_to_user=False
        )

        self.verification_prompt = ver
    
class DeliveryTemplateInstruction(TemplateInstruction):

    def __init__(self, template: Dict, timestamp: int = 0) -> None:
        context = """
        You will have access to a dict with add and remove field. 
        You must generate an instruction that inform the robot hat the default recipe for its packaging has changed. 
        The add field represent element that need to be added to the recipe, remove element represent elements that need to be removed from the recipe.
        If an element is present in both, you can safely ignore it. If there is x time the same element in the same field, you can just tell the robot x element. 
        You are only giving a constraint, you MUST NOT ask to launch a cycle with this recipe, just update it for future cycle.
        Here are some exemple 'add x to your recipe for future cycle', 'remove y and add x from your recipe now' ...
        """
        super().__init__(template, context, timestamp)

class BenchDeliveryTask(BaseTask):
    """
    Class for the Delivery Benchmark Task. The idea is to have a model which must follow a recipe to complete a delivery box.
    It must reason about multiple constraints.
    """

    name : str = "Make devlivery [benchmark]"

    randomized_config_path = str(Path(__file__).resolve().parent.parent / "delivery.yaml")

    env_id = "DeliveryBase-v1"
    Tools_cls = CycleTool

    styles = [
        TaskStyle.CONSTRAINED
    ]

    all_task_attributes = att

# launch_cycle(manufacturing_order="A47",delivery_number=10,recipe=["coca"|"brets"])

    def __init__(
            self,
            stages : List[Dict] = [
                {"constraint" : "Consider the default recipe as 1 coca and 1 brets"},
                {"cycle": {
                    "instruction": "Can you launch a cycle for 10 deliveries under manufacturing order A47",
                    "delivery" : 10,
                    "manufacturing_order" : "A47",
                    "recipe": ["coca","brets"]
                }},
                {"cycle": {
                    "instruction": "Can you launch a cycle for 84 deliveries under manufacturing order K24",
                    "delivery" : 84,
                    "manufacturing_order" : "K24",
                    "recipe": ["coca","brets"]
                }}
            ]
        ) -> None:
        """
        To create this stage you can define stages, a list of Dict. Each dict could either contains a "constraint" elements or a "cycle".
        Like using constraints allows to define an instruction containing a constraint that must respect the model.
        In cycle you must define instruction (the textual query for the model), delivery (the number of delivery), manufacturing order (the manufacturing order) and finally the recipe (a list of object to put in).
        You can compose this as you want, alternating constraints and cycle or just sending multiple cycle query in a row. More you have of stage, more it's complicated.
        """
        super().__init__()

        self.stages = []

        for stage in stages:
            if "constraint" in stage:
                if isinstance(stage["constraint"],str):
                    self.stages.append(ConstraintBaseStage(reset_at_end=False, instruction=UserInstruction(stage["constraint"])))
                else:
                    self.stages.append(ConstraintBaseStage(DeliveryTemplateInstruction(stage['constraint'])))
            else:
                s = stage['cycle']
                self.stages.append(
                    CycleStage(
                        recipe=s["recipe"],
                        manufacturing_order=s["manufacturing_order"],
                        deliveries=s["delivery"],
                        instruction=s["instruction"]
                    )
                )

        if len(self.stages) > 5:
            self.approximal_difficulty = "Hard"
        if len(self.stages) < 3:
            self.approximal_difficulty = "Easy"
        else:
            self.approximal_difficulty = "Medium"

class EvolvingDelivery(BenchDeliveryTask):
    """
    The EvolvingDelivery task allows to define a recipe and ask for a cycle and repeat this as much as needed.
    If hard_mode is set, instead of redefining a recipe it will add or remove small element from it.
    noise_in_between allows to define how many cycle in each loop
    """

    def __init__(self, loop : int = 4, hard_mode : bool = False, noise_in_between : int = 1) -> None:
        
        random_recipe = random.sample(att["product_type"],k=random.randint(1,len(att["product_type"])))

        stages : List[Dict] = [{"constraint":f"Here is the default recipe for all future packaging : {','.join(random_recipe)}"}]
        for _ in range(loop):
            for _ in range(noise_in_between):
                nb = random.randint(1,99)
                mn = craft_random_manu_order(4)
                stages.append({
                    "cycle" : {
                        "instruction" : f"I need {nb} cycle right now. Use manufacturing order {mn}",
                        "delivery": nb,
                        "manufacturing_order": mn,
                        "recipe": random_recipe
                    }
                })
            if hard_mode:
                d = _dynamic_evolve(random_recipe)
                random_recipe = d.pop("final")
                stages.append({"constraint":d})
            else:
                random_recipe = random.sample(att["product_type"],k=random.randint(1,len(att["product_type"])))
                stages.append({"constraint":f"Hey, please update your recipe with {','.join(random_recipe)}"})

        super().__init__(stages)

def _dynamic_evolve(cur_recipe : List[str], max_nb : int = 2, nb_of_modif : int = 1) -> Dict:
    possible = []
    added = []
    removed = []
    c = cur_recipe.copy()
    for p in att['product_type']:
        n = cur_recipe.count(p)
        possible.extend([p]*(max_nb-n))
    
    for _ in range(nb_of_modif):
        if len(c) <= 1:
            ac = 1
        elif len(possible) <= 1:
            ac = 0
        else:   
            ac = random.randint(0,1)
        
        if ac == 1:
            select = random.choice(possible)
            possible.remove(select)
            added.append(select)
        else:
            select = random.choice(c)
            c.remove(select)
            removed.append(select)
    c.extend(added)

    return {
        "add" : added,
        "remove": removed,
        "final" : c
    }

    
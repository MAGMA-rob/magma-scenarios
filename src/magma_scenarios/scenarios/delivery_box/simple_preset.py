# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, List, Dict, Tuple
from pathlib import Path

from magma_core.base.tasks import BaseTask, BaseBenchmarkTask
from magma_core.base.stage import ConstraintBaseStage
from magma_core.base.tasks_style import TaskStyle

from .tools.simple_tool import CycleTool
from .attributes import att
from .delivery_stages import CycleStage

class BenchDeliveryTask(BaseBenchmarkTask):
    """
    Class for the Delivery Benchmark Task. The idea is to have a model which must follow a recipe to complete a delivery box.
    It must reason about multiple constraints.
    """

    name : str = "Make devlivery [benchmark]"

    randomized_config_path = str(Path(__file__).resolve().parent / "delivery.yaml")

    env_id = "DeliveryBase-v1"
    Tools_cls = CycleTool

    styles = [
        TaskStyle.CONSTRAINED
    ]

    all_task_attributes = att

class SimplePreset(BaseTask):
    """
    Class for the Delivery Benchmark Task. The idea is to have a model which must follow a recipe to complete a delivery box.
    It must reason about multiple constraints.
    """

    name : str = "Make devlivery"

    randomized_config_path = str(Path(__file__).resolve().parent / "delivery.yaml")

    env_id = "DeliveryBase-v1"
    Tools_cls = CycleTool

    styles = [
        TaskStyle.CONSTRAINED
    ]

    all_task_attributes = att

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
                    self.stages.append(ConstraintBaseStage(constraint=stage["constraint"],memory=[],attributes=att))
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



    
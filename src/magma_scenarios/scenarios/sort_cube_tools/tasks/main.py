# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tasks import BaseTask, BaseBenchmarkTask
from magma_core.base.tasks_style import TaskStyle

import torch, random
from typing import List, Dict

from ..stages.main import Constraint, CubeInBox, CubeOnAnother
from ..tools.tool import Tool

random_noise_instruction = [
    "Hello can you store {obj} in white box?",
    "Please put {obj} in the box.",
    "Can you put {obj} in box?",
    "Send {obj} to box",
    "I need {obj} in the bow right now",
    "Hey, send {obj} to box please"
]

tower_requests = [
    "I want a tower.",
    "Build a tower.",
    "Please build a tower.",
    "Can you build a tower?",
    "I want you to build a tower.",
    "Can you build a tower as I explained earlier?",
    "Build the tower we discussed before.",
    "Do the tower assembly I described earlier.",
    "Go ahead and build a tower.",
    "Start building the tower now.",
    "Proceed with the tower construction.",
    "Construct a cube tower.",
    "Build a cube tower as instructed.",
    "Build the tower according to my instructions.",
    "Recreate the tower I described earlier."
]

tower_instructions = [
    "To build a tower, you must stack the {color_of_top} cube on the {color_of_bottom}.",
    "When building the tower, place the {color_of_top} cube on top of the {color_of_bottom} one.",
    "The tower should be assembled by stacking the {color_of_top} cube over the {color_of_bottom} cube.",
    "Build the tower so that the {color_of_top} cube is positioned on the {color_of_bottom}.",
    "For the tower, make sure the {color_of_top} cube goes on top of the {color_of_bottom} cube.",

    "Hello, when I ask you to build a tower, you should stack the {color_of_top} cube on the {color_of_bottom}.",
    "Just so you know, a tower means placing the {color_of_top} cube on top of the {color_of_bottom} cube.",
    "Whenever I mention building a tower, I expect the {color_of_top} cube to be stacked on the {color_of_bottom}.",
    "If I ask for a tower, build it by putting the {color_of_top} cube above the {color_of_bottom} one.",
    "From now on, building a tower means stacking the {color_of_top} cube on the {color_of_bottom} cube."
]

class SortCubeBench(BaseBenchmarkTask): #This one is for benchmark only
    """
    Task where the model must stack or store cube in a box.
    """

    approximal_difficulty = "Easy"
    name = "Sort Cubes BENCHMARK"
    env_id = "SortCubesRGB"
    Tools_cls = Tool

    def __init__(self) -> None:
        super().__init__()

class CubeTower(BaseTask):
    """The goal of this tool is to put in memory the explication on how to do a tower and ask the model to do a tower."""

    styles = [
        TaskStyle.COLOR_REASONING,
        TaskStyle.LONG_STAGE,
    ]

    env_id = "SortCubesRGB"

    Tools_cls = Tool

    name = "Build Tower"

    def __init__(
            self,
            color_of_top : str = "blue", color_of_bottom : str = "red",
            number_of_noise_step : int = 2) -> None:
        super().__init__()
        tower_constraint = random.choice(tower_instructions).format(color_of_top=color_of_top, color_of_bottom=color_of_bottom)
        self.stages = [Constraint(tower_constraint, [])]
        cubes = ["red_cube", "blue_cube", "green_cube"]

        for i in range(number_of_noise_step):
            obj_name = random.choice(cubes)
            instruction = random.choice(random_noise_instruction).format(obj=obj_name)
            self.stages.append(CubeInBox(reset_at_end=True,obj_name=obj_name,instruction=instruction,mem=[]))

        tower_ins = random.choice(tower_requests)
        self.stages.append(CubeOnAnother(False,f"{color_of_top}_cube",f"{color_of_bottom}_cube",tower_ins,[]))

        if number_of_noise_step > 4:
            self.approximal_difficulty = "Medium"
        else:
            self.approximal_difficulty = "Easy"
    

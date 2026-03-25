# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
# Author: Arthur TANNEAU
from magma_core.base.data_structures import UserInstruction, EmptyInstruction, Instruction, Log, Situation
from magma_core.base.tasks import BaseTask
from magma_core.base.tasks_style import TaskStyle

import random
from typing import List
from pathlib import Path

from .tool import Tool
from .helper import attributes
from .button_stages import PressMultipleButton, PressButton, PressConstraintStage

class ButtonPressNoOrdering(BaseTask):
    """
    Task to press multiple buttons without explicit ordering.
    You can specify any button betwen sw0 and sw4.
    You can ask to press buttons according to the result of an addition, even, odd, some specific etc...
    Difficulty range is easy to medium
    """

    name : str = "Pressing button without ordering"
    env_id = "PressButtonBasic-v1"
    randomized_config_path = str(Path(__file__).parent.joinpath("press_button_cfg.yaml"))
    Tools_cls = Tool

    styles = [
        TaskStyle.LONG_STAGE
    ]
    all_task_attributes = attributes
    
    approximal_difficulty = "Easy"

    def __init__(self, button_names : List[str] = ["sw2","sw1","sw2"], instruction : str = "Can you press the button 2 two times and one time the button one") -> None:
        """
        You can set an instruction and the list of button that need to be pressed according to your instruction.
        """
        super().__init__()
        main_instruction = UserInstruction(instruction)
        self.stages = []
        a = ["sw0","sw1","sw2"]
        for i in range(1,len(a)+1):
            self.stages.append(
                PressMultipleButton(i,button_names,main_instruction,False)
            )
            self.stages[-1].situation.attributes = {"objects":a}
            main_instruction = EmptyInstruction()
    

class ButtonPressOrdered(BaseTask):
    """
    Task to press multiple buttons with an explicit order communicated by the user in a instruction.
    You can specify any button betwen sw0 and sw4.
    You can ask to press multiple time the same buttons, some specific orders.
    Difficulty range from easy to medium
    """

    name : str = "Pressing button without ordering"
    env_id = "PressButtonBasic-v1"
    randomized_config_path = str(Path(__file__).parent.joinpath("press_button_cfg.yaml"))
    Tools_cls = Tool

    styles = [
        TaskStyle.LONG_STAGE
    ]
    all_task_attributes = attributes

    approximal_difficulty = "Medium"
    

    def __init__(self, sequence_of_buttons : List[str] = ["sw0","sw0","sw4"], instruction : str = "Please press two times sw0 then sw4?") -> None:
        """
        You need to define an instruction and a sequence of button that need to be pressed. The instruction should correspond exactly to the sequence asked.
        You can ask for multiple press per buttons.
        Difficulty is associated with the length of the list : <5 EASY , 5+ Medium.
        """
        super().__init__()
        main_instruction = UserInstruction(instruction)
        self.stages = []
        for i,btn in enumerate(sequence_of_buttons):
            self.stages.append(
                PressButton(btn,main_instruction,i==len(sequence_of_buttons)-1)
            )
            main_instruction = EmptyInstruction()

class ButtonPressPreset2(BaseTask):

    name : str = "Complex button pressing instruction"
    env_id = "PressButtonBasic-v1"
    randomized_config_path = str(Path(__file__).parent.joinpath("press_button_cfg.yaml"))
    Tools_cls = Tool

    styles = [
        TaskStyle.CONSTRAINED,
        TaskStyle.LONG_STAGE
    ]

    all_task_attributes = attributes

    approximal_difficulty = "Medium"

    def __init__(self, nb_of_stage : int = 8) -> None:
        btn = attributes["objects"].copy()
        random.shuffle(btn)
        att = btn[:-2]

        self.stages = []
        for i in range(nb_of_stage):
            if random.randint(0,1) >= 1:
                btns = random.sample(att,k=2)
                self.stages.extend([
                    PressButton(btns[0], UserInstruction(f"Ok press {btns[0]}, then {btns[1]}"), last=False),
                    PressButton(btns[1], EmptyInstruction(), last=True),
                ])
                self.stages[-2].situation.attributes = {"objects":att}
                self.stages[-1].situation.attributes = {"objects":att}
            else:
                b = random.choice(att)
                self.stages.append(PressButton(b, UserInstruction(f"Let's press {b} now"), last=True))    
            self.stages[-1].situation.attributes = {"objects":att}
            
        super().__init__()

class ButtonPressPreset1(BaseTask):

    name : str = "Button press constraint"
    env_id = "PressButtonBasic-v1"
    randomized_config_path = str(Path(__file__).parent.joinpath("press_button_cfg.yaml"))
    Tools_cls = Tool

    styles = [
        TaskStyle.CONSTRAINED,
        TaskStyle.LONG_STAGE
    ]

    all_task_attributes = attributes

    approximal_difficulty = "Medium"

    def __init__(self, difficulty : int = 2, constraint_overriding : bool = False) -> None:
        btn = attributes["objects"].copy()[:-1]
        random.shuffle(btn)
        first_btn = btn.pop(0)
        past_btn = []
        self.stages : List = [
            PressButton(instruction=UserInstruction(f"Please press {btn[0]}"),button=btn[0], last=True),
            PressButton(instruction=UserInstruction(f"Please press {btn[1]}"),button=btn[1], last=True),
            PressConstraintStage(f"Each time you are asked to press a button, press {first_btn} first")
            ]
        random.shuffle(btn)
        for i in range(difficulty):
            if len(btn) == 0:
                break
            btn_to_press = btn.pop(0)
            past_btn.append(first_btn)
            self.stages.extend([
                PressButton(instruction=UserInstruction(f"Please press {btn_to_press}"),button=first_btn),
                PressButton(instruction=EmptyInstruction(),button=btn_to_press, last=True)
            ])

            if constraint_overriding:
                self.stages.extend([
                    PressConstraintStage(f"Ok let's forget the order to press {first_btn} first, okay?"),
                    PressButton(instruction=UserInstruction(f"Please press {btn_to_press} now"),button=btn_to_press, last=True)
                ])
                if i != difficulty-1:
                    first_btn = random.choice(past_btn)
                    self.stages.extend([
                        PressConstraintStage(f"Hey now you need to press {first_btn} before pressing any buttons that i ask, understood?"),
                    ])
            
        super().__init__()
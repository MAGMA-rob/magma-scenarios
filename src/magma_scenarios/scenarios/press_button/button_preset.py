# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
# Author: Arthur TANNEAU
from magma_core.simulation.data_structures import UserInstruction, EmptyInstruction, SituationInit
from magma_core.simulation.tasks import BaseTask
from magma_core.simulation.stage import ConstraintBaseStage

import random
from typing import List
from pathlib import Path

from .tool import Tool
from .helper import attributes
from .button_stages import PressMultipleButton, PressButton

class BaseButton(BaseTask):

    maniskill_env_id = "PressButtonBasic-v1"
    randomized_config_path = str(Path(__file__).parent.joinpath("press_button_cfg.yaml"))
    Tools_cls = Tool
    all_task_attributes = attributes
    situation_init = SituationInit(
        attributes=attributes,
        all_task_attributes=attributes
    )


class ButtonPressNoOrdering(BaseButton):
    """
    Task to press multiple buttons without explicit ordering.
    You can specify any button betwen sw0 and sw4.
    You can ask to press buttons according to the result of an addition, even, odd, some specific etc...
    Difficulty range is easy to medium
    """

    name : str = "Pressing button without ordering"

    def __init__(
            self,
            button_names : List[str] = ["sw2","sw1"],
            instruction : str = "Can you press the button 2 and button one"
        ) -> None:
        """
        You can set an instruction and the list of button that need to be pressed according to your instruction.
        """
        super().__init__()
        self.situation_init.attributes = {"objects": ["sw0","sw1","sw2"]}
        main_instruction = UserInstruction(instruction)
        self.stages = []
        for i in range(1,len(button_names)+1):
            self.stages.append(
                PressMultipleButton(i,button_names,main_instruction,False)
            )
            main_instruction = EmptyInstruction()


class ButtonPressOrdered(BaseButton):
    """
    Task to press multiple buttons with an explicit order communicated by the user in a instruction.
    You can specify any button betwen sw0 and sw4.
    You can ask to press multiple time the same buttons, some specific orders.
    Difficulty range from easy to medium
    """

    name : str = "Pressing button ordered"
    

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

class ButtonPressPreset2(BaseButton):

    name : str = "Complex button pressing instruction"

    def __init__(self, nb_of_stage : int = 8) -> None:
        btn = attributes["objects"].copy()
        random.shuffle(btn)
        att = btn[:-2]
        self.situation_init.attributes = {"objects":att}

        self.stages = []
        for i in range(nb_of_stage):
            if random.randint(0,1) >= 1:
                btns = random.sample(att,k=2)
                self.stages.extend([
                    PressButton(btns[0], UserInstruction(f"Ok press {btns[0]}, then {btns[1]}"), last=False),
                    PressButton(btns[1], EmptyInstruction(), last=True),
                ])
            else:
                b = random.choice(att)
                self.stages.append(PressButton(b, UserInstruction(f"Let's press {b} now"), last=True))    
            
        super().__init__()

class ButtonPressPreset1(BaseButton):

    name : str = "Button press constraint"

    def __init__(self, difficulty : int = 2, constraint_overriding : bool = False) -> None:
        btn = attributes["objects"].copy()[:-1]
        random.shuffle(btn)
        first_btn = btn.pop(0)
        past_btn = []
        self.stages : List = [
            PressButton(instruction=UserInstruction(f"Please press {btn[0]}"),button=btn[0], last=True),
            PressButton(instruction=UserInstruction(f"Please press {btn[1]}"),button=btn[1], last=True),
            ConstraintBaseStage(f"Each time you are asked to press a button, press {first_btn} first")
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
                    ConstraintBaseStage(f"Ok let's forget the order to press {first_btn} first, okay?"),
                    PressButton(instruction=UserInstruction(f"Please press {btn_to_press} now"),button=btn_to_press, last=True)
                ])
                if i != difficulty-1:
                    first_btn = random.choice(past_btn)
                    self.stages.extend([
                        ConstraintBaseStage(f"Hey now you need to press {first_btn} before pressing any buttons that i ask, understood?"),
                    ])
            
        super().__init__()
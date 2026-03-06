# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# Arthur TANNEAU
from magma_core.base.data_structures import UserInstruction, EmptyInstruction, Instruction, Log, Situation
from magma_core.base.tasks import BaseTask, BaseTaskStage, ConstraintBaseStage
from magma_core.base.tasks_style import TaskStyle

import torch, random
from typing import List, Dict, Tuple
from pathlib import Path

from .tool import Tool, BTN_STROKE

def tensor_is_button_pressed(btn_translation) -> torch.Tensor:
    button_STROKE_LIMIT = -BTN_STROKE/2 * torch.ones(btn_translation.size())
    return (btn_translation < button_STROKE_LIMIT)

attributes = {"objects": ["sw0","sw1","sw2","sw3","sw4"]}

class PressMultipleButton(BaseTaskStage):

    target_steps = 1
    acceptance_steps = 1

    def __init__(self, n : int, all_button : List[str], instruction : Instruction, reset_at_end: bool = False) -> None:
        super().__init__(reset_at_end, f"The goal of this stage is to have {n} button pressed from the list : {all_button}")
        self.button_names = all_button
        self.nb = n
        self.situation = Situation(
            memory=[],
            preserved_memory_indices=[],
            attributes=attributes,
            instruction=instruction,
            flag_answer_to_user= n == len(all_button)
        )

    
    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        """ Check if only the asked button is pressed """
        device = obs["extra"][self.button_names[0]].device
        batch_size = obs["extra"][self.button_names[0]].shape[0]

        pressed = torch.zeros(batch_size, device=device, dtype=torch.int32)
        err = torch.zeros(batch_size, device=device, dtype=torch.bool)
        for obj_name, pose in obs["extra"].items():
            if "sw" not in obj_name:
                continue
            is_pressed = tensor_is_button_pressed(pose[:, -1])

            if obj_name in self.button_names:
                pressed += is_pressed.int()
            else:
                err |= is_pressed

        out = (pressed == self.nb).int()
        out[err] = -1
        return out.int()
    
class ConstraintStage(ConstraintBaseStage):

    def __init__(self, constraint: str, memory: List[str]) -> None:
        super().__init__(constraint, memory, attributes)

class PressButton(BaseTaskStage):

    target_steps = 1
    acceptance_steps = 1

    def __init__(self, button : str, instruction : Instruction, last : bool = False) -> None:
        super().__init__(True, f"The goal of this stage is to press {button}")
        self.button = button
        self.situation = Situation(
            memory=[],
            preserved_memory_indices=[],
            attributes=attributes,
            instruction=instruction,
            flag_answer_to_user= last
        )
    
    def verif_env_completion(self, obs: Dict) -> torch.Tensor:
        """ Check if only the asked button is pressed """
        return tensor_is_button_pressed(obs["extra"][self.button][:, -1])
        
    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        if len(stage_log) == 0: return 0
        if stage_log[-1].content == self.button: return 1
        return -1

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

class ConstrainedButtonPress(BaseTask):
    """
    Task where the robot must press multiple buttons asked by the user while respecting some ordering constraint said previosuly.
    The diffulcty is Medium.
    """

    name : str = "Pressing button without ordering"
    env_id = "PressButtonBasic-v1"
    randomized_config_path = str(Path(__file__).parent.joinpath("press_button_cfg.yaml"))
    Tools_cls = Tool

    styles = [
        TaskStyle.CONSTRAINED,
        TaskStyle.LONG_STAGE
    ]

    all_task_attributes = attributes

    approximal_difficulty = "Medium"
    
    def __init__(
            self,
            stages : List[Tuple[str,str,List]] = [
                # ("constraint" , "You must always press even buttons before odd", []),
                ("cycle", "Can you press sw1 and sw0",[["sw0","sw1"]]),
                ("cycle", "Can you press sw3 now",[["sw3"]]),
                ("cycle", "Please let's press button 1 and 2",[["sw2","sw1"]]),
                ("constraint" , "Now, When i ask you to press one buttons, I want you to press sw4 before pressing my buttons.", []),
                ("cycle", "Please let's press button 1",[["sw4"],["sw1"]]),
                ("cycle", "Okay do it again",[["sw4"],["sw1"]])
            ]
        ) -> None:
        """
        Here you need to build a list of differents stages instruction. An instruction is a tuple of three elements.
        The first could either be "constraint or "cycle". 
        "constraint" indicates that you are giving an information to the robot that are related to the next cycle.
        "cycle" is when you ask for a specific action from the robot. In that case, pressing buttons
        The second element is the explicit instruction given to the model. It's the sentence the user say.
        For a "constraint" the third element of the tuple is just an empty list. But for a "cycle" it's a list of grouped buttons that should be pressed.
        Meaning that each buttons in the same inner list could be pressed in any order. But each buttons of the same group should be pressed before passing to the next one.
        Be aware that the list of button should respect instruction and constraint you gave earlier.
        """
        super().__init__()
        self.stages = []
        for s in stages:
            if s[0] == "constraint":
                self.stages.append(ConstraintStage(s[1],[]))
            else:
                for group in s[2]:
                    instruction = UserInstruction(s[1])
                    for i in range(len(group)):
                        self.stages.append(PressMultipleButton(i+1, group, instruction, i == len(group)-1))
                        instruction = EmptyInstruction()

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
            ConstraintStage(f"Each time you are asked to press a button, press {first_btn} first",[])
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
                    ConstraintStage(f"Ok let's forget the order to press {first_btn} first, okay?",[]),
                    PressButton(instruction=UserInstruction(f"Please press {btn_to_press} now"),button=btn_to_press, last=True)
                ])
                if i != difficulty-1:
                    first_btn = random.choice(past_btn)
                    self.stages.extend([
                        ConstraintStage(f"Hey now you need to press {first_btn} before pressing any buttons that i ask, understood?",[]),
                    ])
            
        super().__init__()

# class ButtonPressPreset2(BaseTask):

#     name : str = "Complex button pressing instruction"
#     env_id = "PressButtonBasic-v1"
#     randomized_config_path = str(Path(__file__).parent.joinpath("press_button_cfg.yaml"))
#     Tools_cls = Tool

#     styles = [
#         TaskStyle.CONSTRAINED,
#         TaskStyle.LONG_STAGE
#     ]

#     all_task_attributes = attributes

#     approximal_difficulty = "Medium"

#     def __init__(self) -> None:
#         btn = attributes["objects"].copy()[:-2]
#         att = btn.copy()
#         random.shuffle(btn)

#         instruction = TemplateInstruction({"order":btn}, context="""
#                                           You have access to a dict with an order keys. This list correspond to the strict order that the robot must follow to press buttons.
#                                           You must generate a strict instruction that ask to press buttons in this order.
#                                           e.g ['btn2','btn1','btn3'] -> 'I want you to press first btn2, then btn1 and finally btn3'.
#                                           """)
#         self.stages = []
#         for i in range(len(btn)):
#             self.stages.append(PressButton(btn[i], instruction, last=(i==len(btn)-1)))
#             self.stages[-1].situation.attributes = {"objects":att}
#             instruction = EmptyInstruction()
            
#         super().__init__()

# class ButtonConstraint1(ButtonPressNoOrder):
#     button_names = ["sw0","sw2","sw4"]
#     def _build_init_elements(self):
#         self.memory = [
#         "You are in charge of pressing one or many buttons knowing their names with a given pythonic list.",
#         "Always press first even buttons, others after."
#         ]
#         self.attributes = {"objects": []}
#         for i in range(5):
#             self.attributes["objects"].append("sw" + str(i))
#         self.instruction = "Can you press all buttons please?"

# class ButtonConstraint2(ButtonPressNoOrder):

#     def __init__(self, target_steps: int = 2, acceptance_steps: int = 0, **kwargs):
#         super().__init__(target_steps, acceptance_steps, **kwargs)

#     button_names = ["sw3","sw1"]
#     def _build_init_elements(self):
#         self.memory = [
#         "You are in charge of pressing one or many buttons knowing their names with a given pythonic list.",
#         "Always press first even buttons, others after.",
#         "The user asked to press all buttons. Buttons already pressed : sw0, sw4",
#         "I am pressing sw2"
#         ]
#         self.attributes = {"objects": []}
#         for i in range(5):
#             self.attributes["objects"].append("sw" + str(i))
#         self.instruction = "{'infos': 'press_button succeed : You have the sw2 button pressed.'}"

# class ButtonConstraint3(ButtonPress):
#     button_names = ["sw3","sw4"]

#     def __init__(self, target_steps: int = 2, acceptance_steps: int = 1, **kwargs):
#         super().__init__(target_steps, acceptance_steps, **kwargs)

#     def _build_init_elements(self):
#         self.memory = [
#         "You are in charge of pressing one or many buttons knowing their names with a given pythonic list.",
#         "Button 3 must always be pressed first",
#         "Button 4 must be pressed before 0, 1 and 2"
#         ]
#         self.attributes = {"objects": []}
#         for i in range(5):
#             self.attributes["objects"].append("sw" + str(i))
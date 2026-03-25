# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# Arthur TANNEAU
import random
from typing import List

from magma_core.base.data_structures import EmptyInstruction, UserInstruction
from magma_core.base.stage import BaseTaskStage
from magma_core.base.state import TaskState
from magma_core.base.user_request import BaseConstraintRequest, BaseRequest

from .button_constraints import (
    BUTTON_SEQUENCE_PREFIX_KEY,
    ButtonPrecedenceConstraint,
    ButtonSequencePrefixConstraint,
    ForgetButtonRulesConstraint,
    available_forget_targets,
    available_buttons,
    button_list_to_text,
    can_add_rule,
    count_active_rules,
    has_any_button_rule,
    resolve_order_with_rules,
    resolve_prefix_only_order,
    valid_group_candidates,
    valid_parity_candidates,
)
from .button_stages import PressButton

MAX_SIMULTANEOUS_RULES = 3


def _sample_requested_buttons(state: TaskState, max_nb_btn: int) -> List[str]:
    """Sample a non-empty set of distinct buttons from the current task state."""
    buttons = available_buttons(state)
    if not buttons:
        raise RuntimeError("Failed to sample buttons because the task state does not expose any button")

    nb_buttons = random.randint(1, min(max_nb_btn, len(buttons)))
    return random.sample(buttons, k=nb_buttons)


def _build_exact_order_instruction(buttons: List[str]) -> UserInstruction:
    """Instruction where the user explicitly states the expected order."""
    if len(buttons) == 1:
        return UserInstruction(f"Please press {buttons[0]}.")
    return UserInstruction(
        f"Please press the following buttons in this exact order: {button_list_to_text(buttons)}."
    )


def _build_unordered_instruction(buttons: List[str]) -> UserInstruction:
    """Instruction where the user only lists buttons, without giving an order."""
    if len(buttons) == 1:
        return UserInstruction(f"Please press {buttons[0]}.")
    return UserInstruction(f"Please press {button_list_to_text(buttons)}.")


def _build_press_stages(button_sequence: List[str], instruction: UserInstruction) -> List[BaseTaskStage]:
    """Convert a resolved button sequence into one PressButton stage per button."""
    stages = []
    current_instruction = instruction

    for index, button_name in enumerate(button_sequence):
        stages.append(
            PressButton(
                button=button_name,
                instruction=current_instruction,
                last=(index == len(button_sequence) - 1),
            )
        )
        current_instruction = EmptyInstruction()

    return stages


class GiveButtonGroupOrderRequest(BaseConstraintRequest):
    """Sample a permanent precedence rule between two button groups."""

    def __init__(self, max_buttons_per_group: int = 2, max_active_rules: int = MAX_SIMULTANEOUS_RULES) -> None:
        super().__init__()
        self.max_buttons_per_group = max_buttons_per_group
        self.max_active_rules = max_active_rules

    def sampling_weight(self, state: TaskState) -> float:
        if not can_add_rule(state, self.max_active_rules):
            return 0
        if len(available_buttons(state)) < 2:
            return 0
        return 1 if valid_group_candidates(state, self.max_buttons_per_group) else 0

    def initialize_constraints(self, state: TaskState):
        candidates = valid_group_candidates(state, self.max_buttons_per_group)
        if not candidates:
            raise RuntimeError(f"Failed to sample a valid group order in {self.__class__.__name__}")

        first_group, second_group = random.choice(candidates)
        self.constraints = [
            ButtonPrecedenceConstraint(first_group, second_group, rule_name="group-order")
        ]
        self.constraint_msg = (
            f"From now on, press {button_list_to_text(first_group)} "
            f"before {button_list_to_text(second_group)} whenever they are both requested."
        )


class GiveEvenOddOrderRequest(BaseConstraintRequest):
    """Sample an even-first or odd-first permanent precedence rule."""

    def __init__(self, max_active_rules: int = MAX_SIMULTANEOUS_RULES) -> None:
        super().__init__()
        self.max_active_rules = max_active_rules

    def sampling_weight(self, state: TaskState) -> float:
        if not can_add_rule(state, self.max_active_rules):
            return 0
        return 1 if valid_parity_candidates(state) else 0

    def initialize_constraints(self, state: TaskState):
        candidates = valid_parity_candidates(state)
        if not candidates:
            raise RuntimeError(f"Failed to sample a valid parity order in {self.__class__.__name__}")

        first_group, second_group, label = random.choice(candidates)
        self.constraints = [
            ButtonPrecedenceConstraint(first_group, second_group, rule_name=label)
        ]

        if label == "even-first":
            self.constraint_msg = (
                "From now on, when both even and odd buttons are requested, "
                "press all even buttons before the odd ones."
            )
        else:
            self.constraint_msg = (
                "From now on, when both even and odd buttons are requested, "
                "press all odd buttons before the even ones."
            )


class GiveSequencePrefixRequest(BaseConstraintRequest):
    """Sample a permanent rule of the form: always press one button first."""

    def __init__(self, max_active_rules: int = MAX_SIMULTANEOUS_RULES) -> None:
        super().__init__()
        self.max_active_rules = max_active_rules

    def sampling_weight(self, state: TaskState) -> float:
        if not can_add_rule(state, self.max_active_rules):
            return 0
        if len(available_buttons(state)) == 0:
            return 0
        if state.relations.get(BUTTON_SEQUENCE_PREFIX_KEY, {}):
            return 0
        return 1

    def initialize_constraints(self, state: TaskState):
        buttons = available_buttons(state)
        if not buttons:
            raise RuntimeError(f"Failed to sample a prefix button in {self.__class__.__name__}")

        button_name = random.choice(buttons)
        self.constraints = [ButtonSequencePrefixConstraint(button_name)]
        self.constraint_msg = (
            f"From now on, press {button_name} before the sequence of buttons that I ask you to press."
        )


class ForgetButtonRulesRequest(BaseConstraintRequest):
    """Clear all accumulated permanent press-button rules."""

    def sampling_weight(self, state: TaskState) -> float:
        if not has_any_button_rule(state):
            return 0
        if count_active_rules(state) >= MAX_SIMULTANEOUS_RULES:
            return 3
        return 1

    def initialize_constraints(self, state: TaskState):
        targets = available_forget_targets(state)
        if not targets:
            raise RuntimeError(f"Failed to sample a forget target in {self.__class__.__name__}")

        mode, button_name = random.choice(targets)
        self.constraints = [ForgetButtonRulesConstraint(mode=mode, button_name=button_name)]

        if mode == "all":
            self.constraint_msg = "Forget all the permanent button-order rules I gave you before."
        elif mode == "prefix":
            self.constraint_msg = "Forget all prefix rules."
        elif mode == "precedence-source":
            self.constraint_msg = f"Forget to press {button_name} before others."
        else:
            raise RuntimeError(f"Unknown forget mode sampled: {mode}")


class AskButtonsInExactOrderRequest(BaseRequest):
    """Ask for a random button sequence with an explicit user order.

    Only the prefix rule is applied on top of the sampled order.
    """

    def __init__(self, max_nb_btn: int = 3) -> None:
        super().__init__()
        self.max_nb_btn = max_nb_btn

    def sampling_weight(self, state: TaskState) -> float:
        return 3 if available_buttons(state) else 0

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        requested_buttons = _sample_requested_buttons(state, self.max_nb_btn)
        effective_order = resolve_prefix_only_order(state, requested_buttons)
        instruction = _build_exact_order_instruction(requested_buttons)
        return _build_press_stages(effective_order, instruction)


class AskButtonsRequest(BaseRequest):
    """Ask for random buttons without giving an explicit order.

    The final sequence is derived from all active permanent rules.
    """

    def __init__(self, max_nb_btn: int = 3) -> None:
        super().__init__()
        self.max_nb_btn = max_nb_btn

    def sampling_weight(self, state: TaskState) -> float:
        return 3 if available_buttons(state) else 0

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        requested_buttons = _sample_requested_buttons(state, self.max_nb_btn)
        effective_order = resolve_order_with_rules(state, requested_buttons)
        instruction = _build_unordered_instruction(requested_buttons)
        return _build_press_stages(effective_order, instruction)

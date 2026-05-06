# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# Arthur TANNEAU
import random
from typing import List, Optional

from magma_core.base.data_structures import EmptyInstruction, UserInstruction
from magma_core.base.stage import BaseTaskStage
from magma_core.base.state import TaskState
from magma_core.base.user_request import BaseConstraintRequest, BaseRequest

from .button_constraints import (
    PREFIX_RULE_KIND,
    PRECEDENCE_RULE_KIND,
    ButtonPrecedenceConstraint,
    ButtonSequencePrefixConstraint,
    ForgetButtonRulesConstraint,
    available_buttons,
    available_forget_targets,
    button_list_to_text,
    button_rule_application_seeds,
    button_rule_pending,
    can_add_rule,
    clear_button_rule_pending,
    count_active_rules,
    get_prefix_button,
    has_any_button_rule,
    mark_button_rule_pending,
    pending_button_rule_kind,
    request_uses_pending_button_rule,
    resolve_order_with_rules,
    resolve_prefix_only_order,
    valid_group_candidates,
    valid_parity_candidates,
)
from .button_stages import PressButton

MAX_SIMULTANEOUS_RULES = 2
BASE_ASK_WEIGHT = 3
PENDING_RULE_ASK_WEIGHT = 10
PENDING_RULE_SIDE_REQUEST_WEIGHT = 0.25


def _sample_requested_buttons(state: TaskState, max_nb_btn: int) -> List[str]:
    buttons = available_buttons(state)
    if not buttons:
        raise RuntimeError("Failed to sample buttons because the task state has no button")

    nb_buttons = random.randint(1, min(max_nb_btn, len(buttons)))
    return random.sample(buttons, k=nb_buttons)


def _sample_buttons_to_apply_rule(
    state: TaskState,
    max_nb_btn: int,
    rule_kind: Optional[str] = None,
) -> List[str]:
    buttons = available_buttons(state)
    seeds = [
        seed for seed in button_rule_application_seeds(state, rule_kind)
        if len(seed) <= max_nb_btn
    ]
    if not seeds:
        return _sample_requested_buttons(state, max_nb_btn)

    selected = list(random.choice(seeds))
    remaining_buttons = [button for button in buttons if button not in selected]
    max_extra = min(max_nb_btn, len(buttons)) - len(selected)
    nb_extra = random.randint(0, max_extra)
    selected.extend(random.sample(remaining_buttons, k=nb_extra))
    random.shuffle(selected)
    return selected


def _sample_exact_order_buttons(state: TaskState, max_nb_btn: int) -> List[str]:
    prefix_button = get_prefix_button(state)
    if button_rule_pending(state) and pending_button_rule_kind(state) == PREFIX_RULE_KIND:
        return _sample_buttons_to_apply_rule(state, max_nb_btn, PREFIX_RULE_KIND)

    buttons = available_buttons(state)
    if not buttons:
        raise RuntimeError("Failed to sample buttons because the task state has no button")

    sample_pool = [button for button in buttons if button != prefix_button]
    if not sample_pool:
        sample_pool = buttons

    nb_buttons = random.randint(1, min(max_nb_btn, len(sample_pool)))
    return random.sample(sample_pool, k=nb_buttons)


def _build_press_instruction(buttons: List[str], exact_order: bool) -> UserInstruction:
    if len(buttons) == 1:
        return UserInstruction(f"Please press {buttons[0]}.")
    if exact_order:
        return UserInstruction(
            f"Please press the following buttons in this exact order: {button_list_to_text(buttons)}."
        )
    return UserInstruction(f"Please press {button_list_to_text(buttons)}.")


def _build_press_stages(
    button_sequence: List[str],
    instruction: UserInstruction,
) -> List[BaseTaskStage]:
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


class ButtonRuleRequest(BaseConstraintRequest):
    """Base class for requests that add a button rule needing a follow-up use."""

    rule_kind = PRECEDENCE_RULE_KIND

    def _available_weight(self, state: TaskState) -> float:
        return PENDING_RULE_SIDE_REQUEST_WEIGHT if button_rule_pending(state) else 1

    def apply_request(self, state: TaskState) -> TaskState:
        state = super().apply_request(state)
        if self.constraints:
            mark_button_rule_pending(state, self.rule_kind)
        return state


class GiveButtonGroupOrderRequest(ButtonRuleRequest):
    """Sample a permanent precedence rule between two button groups."""

    rule_kind = PRECEDENCE_RULE_KIND

    def __init__(
        self,
        max_buttons_per_group: int = 2,
        max_active_rules: int = MAX_SIMULTANEOUS_RULES,
    ) -> None:
        super().__init__()
        self.max_buttons_per_group = max_buttons_per_group
        self.max_active_rules = max_active_rules

    def sampling_weight(self, state: TaskState) -> float:
        if not can_add_rule(state, self.max_active_rules):
            return 0
        if len(available_buttons(state)) < 2:
            return 0
        if not valid_group_candidates(state, self.max_buttons_per_group):
            return 0
        return self._available_weight(state)

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


class GiveEvenOddOrderRequest(ButtonRuleRequest):
    """Sample an even-first or odd-first permanent precedence rule."""

    rule_kind = PRECEDENCE_RULE_KIND

    def __init__(
        self,
        max_active_rules: int = MAX_SIMULTANEOUS_RULES,
    ) -> None:
        super().__init__()
        self.max_active_rules = max_active_rules

    def sampling_weight(self, state: TaskState) -> float:
        if not can_add_rule(state, self.max_active_rules):
            return 0
        if not valid_parity_candidates(state):
            return 0
        return self._available_weight(state)

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


class GiveSequencePrefixRequest(ButtonRuleRequest):
    """Sample a permanent rule of the form: always press one button first."""

    rule_kind = PREFIX_RULE_KIND

    def __init__(
        self,
        max_active_rules: int = MAX_SIMULTANEOUS_RULES,
    ) -> None:
        super().__init__()
        self.max_active_rules = max_active_rules

    def sampling_weight(self, state: TaskState) -> float:
        if not can_add_rule(state, self.max_active_rules):
            return 0
        if len(available_buttons(state)) < 2:
            return 0
        if get_prefix_button(state) is not None:
            return 0
        return self._available_weight(state)

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
        if button_rule_pending(state):
            return PENDING_RULE_SIDE_REQUEST_WEIGHT
        if count_active_rules(state) >= MAX_SIMULTANEOUS_RULES - 1:
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
    """Ask for a button sequence with an explicit order.

    Prefix rules can still add a first button. Precedence rules are intentionally
    not applied because the user already gave an exact order.
    """

    def __init__(self, max_nb_btn: int = 3) -> None:
        super().__init__()
        self.max_nb_btn = max_nb_btn
        self._used_pending_rule = False

    def sampling_weight(self, state: TaskState) -> float:
        if not available_buttons(state):
            return 0
        if button_rule_pending(state):
            if pending_button_rule_kind(state) == PREFIX_RULE_KIND:
                return BASE_ASK_WEIGHT
            return PENDING_RULE_SIDE_REQUEST_WEIGHT
        return BASE_ASK_WEIGHT

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        requested_buttons = _sample_exact_order_buttons(state, self.max_nb_btn)
        effective_order = resolve_prefix_only_order(state, requested_buttons)
        self._used_pending_rule = (
            pending_button_rule_kind(state) == PREFIX_RULE_KIND
            and request_uses_pending_button_rule(state, requested_buttons)
        )
        instruction = _build_press_instruction(requested_buttons, exact_order=True)
        return _build_press_stages(effective_order, instruction)

    def apply_request(self, state: TaskState) -> TaskState:
        if self._used_pending_rule:
            clear_button_rule_pending(state)
        return state


class AskButtonsRequest(BaseRequest):
    """Ask for buttons without giving an explicit order.

    When a fresh rule is waiting to be exercised, the sampled buttons include
    the minimal set needed to make that rule affect the resulting sequence.
    """

    def __init__(self, max_nb_btn: int = 3) -> None:
        super().__init__()
        self.max_nb_btn = max_nb_btn
        self._used_pending_rule = False

    def sampling_weight(self, state: TaskState) -> float:
        if not available_buttons(state):
            return 0
        if button_rule_pending(state):
            return PENDING_RULE_ASK_WEIGHT
        if has_any_button_rule(state):
            return 4
        return BASE_ASK_WEIGHT

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        if button_rule_pending(state):
            requested_buttons = _sample_buttons_to_apply_rule(
                state,
                self.max_nb_btn,
                pending_button_rule_kind(state),
            )
        else:
            requested_buttons = _sample_requested_buttons(state, self.max_nb_btn)

        effective_order = resolve_order_with_rules(state, requested_buttons)
        self._used_pending_rule = request_uses_pending_button_rule(state, requested_buttons)
        instruction = _build_press_instruction(requested_buttons, exact_order=False)
        return _build_press_stages(effective_order, instruction)

    def apply_request(self, state: TaskState) -> TaskState:
        if self._used_pending_rule:
            clear_button_rule_pending(state)
        return state

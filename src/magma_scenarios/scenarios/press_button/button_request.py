# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from dataclasses import dataclass
import random
from typing import List, Optional, Tuple

from magma_core.simulation.data_structures import EmptyInstruction, UserInstruction
from magma_core.simulation.stage import BaseTaskStage
from magma_core.simulation.state import TaskState
from magma_scenarios.templates.requests.interact_request import (
    BaseConstraintRequest,
    ConstraintParameters,
)
from magma_core.simulation.requests import BaseRequest

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
    valid_prefix_candidates,
)
from .button_stages import PressButton


MAX_SIMULTANEOUS_RULES = 2
BASE_ASK_WEIGHT = 3
PENDING_RULE_ASK_WEIGHT = 10
PENDING_RULE_SIDE_REQUEST_WEIGHT = 0.25
BUTTON_INTERRUPTION_PROBABILITY = 0.3


@dataclass(frozen=True)
class ButtonInterruption:
    index: int
    button: str
    instruction: str


@dataclass(frozen=True)
class ButtonRequestParameters:
    effective_order: Tuple[str, ...]
    instruction: str
    used_pending_rule: bool
    interruption: Optional[ButtonInterruption]


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
        seed
        for seed in button_rule_application_seeds(state, rule_kind)
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
    if (
        button_rule_pending(state)
        and pending_button_rule_kind(state) == PREFIX_RULE_KIND
    ):
        return _sample_buttons_to_apply_rule(state, max_nb_btn, PREFIX_RULE_KIND)
    buttons = available_buttons(state)
    if not buttons:
        raise RuntimeError("Failed to sample buttons because the task state has no button")
    sample_pool = [button for button in buttons if button != prefix_button]
    if not sample_pool:
        sample_pool = buttons
    nb_buttons = random.randint(1, min(max_nb_btn, len(sample_pool)))
    return random.sample(sample_pool, k=nb_buttons)


def _build_press_instruction(buttons: List[str], exact_order: bool) -> str:
    if len(buttons) == 1:
        return f"Please press {buttons[0]}."
    if exact_order:
        return (
            "Please press the following buttons in this exact order: "
            f"{button_list_to_text(buttons)}."
        )
    return f"Please press {button_list_to_text(buttons)}."


def _sample_button_interruption(
    state: TaskState,
    effective_order: List[str],
) -> Optional[ButtonInterruption]:
    if (
        len(effective_order) < 2
        or random.random() >= BUTTON_INTERRUPTION_PROBABILITY
    ):
        return None
    interruption_candidates = [
        button
        for button in available_buttons(state)
        if button not in effective_order
    ]
    if not interruption_candidates:
        return None
    interruption_button = random.choice(interruption_candidates)
    interruption_index = random.randint(1, len(effective_order) - 1)
    return ButtonInterruption(
        interruption_index,
        interruption_button,
        f"Okay, press {interruption_button} now before resuming your current sequence.",
    )


def _build_press_stages(
    parameters: ButtonRequestParameters,
) -> List[BaseTaskStage]:
    stages = []
    current_instruction = UserInstruction(parameters.instruction)
    interruption = parameters.interruption
    for index, button_name in enumerate(parameters.effective_order):
        stage = PressButton(
            button=button_name,
            instruction=current_instruction,
            last=index == len(parameters.effective_order) - 1,
        )
        stages.append(stage)
        current_instruction = EmptyInstruction()
        if interruption is not None and index == interruption.index - 1:
            interruption_stage = PressButton(
                button=interruption.button,
                instruction=UserInstruction(interruption.instruction),
                last=False,
            )
            interruption_stage.stage_input.linked_to_prev = True
            stages.append(interruption_stage)
    stages[-1].target_tool_calls += 1
    stages[-1].max_tool_calls = max(
        stages[-1].max_tool_calls,
        stages[-1].target_tool_calls,
    )
    return stages


class ButtonRuleRequest(BaseConstraintRequest):
    """Base class for requests that add a rule needing a follow-up use."""

    rule_kind = PRECEDENCE_RULE_KIND

    def _available_weight(self, state: TaskState) -> float:
        return PENDING_RULE_SIDE_REQUEST_WEIGHT if button_rule_pending(state) else 1

    def apply_request(
        self,
        state: TaskState,
        parameters: ConstraintParameters,
    ) -> TaskState:
        state = super().apply_request(state, parameters)
        mark_button_rule_pending(state, self.rule_kind)
        return state


class GiveButtonGroupOrderRequest(ButtonRuleRequest):
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

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        candidates = valid_group_candidates(state, self.max_buttons_per_group)
        if not candidates:
            raise RuntimeError("Failed to sample a valid button group order.")
        first_group, second_group = random.choice(candidates)
        instruction = (
            f"From now on, press {button_list_to_text(first_group)} before "
            f"{button_list_to_text(second_group)} whenever they are both requested."
        )
        return ConstraintParameters(
            [ButtonPrecedenceConstraint(first_group, second_group, "group-order")],
            instruction,
        )


class GiveEvenOddOrderRequest(ButtonRuleRequest):
    rule_kind = PRECEDENCE_RULE_KIND

    def __init__(self, max_active_rules: int = MAX_SIMULTANEOUS_RULES) -> None:
        super().__init__()
        self.max_active_rules = max_active_rules

    def sampling_weight(self, state: TaskState) -> float:
        if not can_add_rule(state, self.max_active_rules):
            return 0
        if not valid_parity_candidates(state):
            return 0
        return self._available_weight(state)

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        candidates = valid_parity_candidates(state)
        if not candidates:
            raise RuntimeError("Failed to sample a valid parity order.")
        first_group, second_group, label = random.choice(candidates)
        if label == "even-first":
            instruction = (
                "From now on, when both even and odd buttons are requested, "
                "press all even buttons before the odd ones."
            )
        else:
            instruction = (
                "From now on, when both even and odd buttons are requested, "
                "press all odd buttons before the even ones."
            )
        return ConstraintParameters(
            [ButtonPrecedenceConstraint(first_group, second_group, label)],
            instruction,
        )


class GiveSequencePrefixRequest(ButtonRuleRequest):
    rule_kind = PREFIX_RULE_KIND

    def __init__(self, max_active_rules: int = MAX_SIMULTANEOUS_RULES) -> None:
        super().__init__()
        self.max_active_rules = max_active_rules

    def sampling_weight(self, state: TaskState) -> float:
        if not can_add_rule(state, self.max_active_rules):
            return 0
        if len(available_buttons(state)) < 2 or get_prefix_button(state) is not None:
            return 0
        if not valid_prefix_candidates(state):
            return 0
        return self._available_weight(state)

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        candidates = valid_prefix_candidates(state)
        if not candidates:
            raise RuntimeError("Failed to sample a prefix button.")
        button_name = random.choice(candidates)
        return ConstraintParameters(
            [ButtonSequencePrefixConstraint(button_name)],
            f"From now on, press {button_name} before the sequence of buttons that I ask you to press.",
        )


class ForgetButtonRulesRequest(BaseConstraintRequest):
    def sampling_weight(self, state: TaskState) -> float:
        if not has_any_button_rule(state):
            return 0
        if button_rule_pending(state):
            return PENDING_RULE_SIDE_REQUEST_WEIGHT
        if count_active_rules(state) >= MAX_SIMULTANEOUS_RULES - 1:
            return 3
        return 1

    def sample_parameters(self, state: TaskState) -> ConstraintParameters:
        targets = available_forget_targets(state)
        if not targets:
            raise RuntimeError("Failed to sample a button-rule forget target.")
        mode, button_name = random.choice(targets)
        if mode == "all":
            instruction = "Forget all the permanent button-order rules I gave you before."
        elif mode == "prefix":
            instruction = "Forget all prefix rules."
        elif mode == "precedence-source":
            instruction = f"Forget to press {button_name} before others."
        else:
            raise RuntimeError(f"Unknown forget mode sampled: {mode}")
        return ConstraintParameters(
            [ForgetButtonRulesConstraint(mode=mode, button_name=button_name)],
            instruction,
        )


class AskButtonsInExactOrderRequest(BaseRequest[ButtonRequestParameters]):
    def __init__(self, max_nb_btn: int = 3) -> None:
        super().__init__()
        self.max_nb_btn = max_nb_btn

    def sampling_weight(self, state: TaskState) -> float:
        if not available_buttons(state):
            return 0
        if button_rule_pending(state):
            if pending_button_rule_kind(state) == PREFIX_RULE_KIND:
                return BASE_ASK_WEIGHT
            return PENDING_RULE_SIDE_REQUEST_WEIGHT
        return BASE_ASK_WEIGHT

    def sample_parameters(self, state: TaskState) -> ButtonRequestParameters:
        requested = _sample_exact_order_buttons(state, self.max_nb_btn)
        effective = resolve_prefix_only_order(state, requested)
        used_pending = (
            pending_button_rule_kind(state) == PREFIX_RULE_KIND
            and request_uses_pending_button_rule(state, requested)
        )
        instruction = _build_press_instruction(requested, exact_order=True)
        interruption = _sample_button_interruption(state, effective)
        return ButtonRequestParameters(
            tuple(effective),
            instruction,
            used_pending,
            interruption,
        )

    def create_stages(
        self,
        state: TaskState,
        parameters: ButtonRequestParameters,
    ) -> List[BaseTaskStage]:
        return _build_press_stages(parameters)

    def apply_request(
        self,
        state: TaskState,
        parameters: ButtonRequestParameters,
    ) -> TaskState:
        if parameters.used_pending_rule:
            clear_button_rule_pending(state)
        return state


class AskButtonsRequest(AskButtonsInExactOrderRequest):
    def sampling_weight(self, state: TaskState) -> float:
        if not available_buttons(state):
            return 0
        if button_rule_pending(state):
            return PENDING_RULE_ASK_WEIGHT
        if has_any_button_rule(state):
            return 4
        return BASE_ASK_WEIGHT

    def sample_parameters(self, state: TaskState) -> ButtonRequestParameters:
        if button_rule_pending(state):
            requested = _sample_buttons_to_apply_rule(
                state,
                self.max_nb_btn,
                pending_button_rule_kind(state),
            )
        else:
            requested = _sample_requested_buttons(state, self.max_nb_btn)
        effective = resolve_order_with_rules(state, requested)
        used_pending = request_uses_pending_button_rule(state, requested)
        instruction = _build_press_instruction(requested, exact_order=False)
        interruption = _sample_button_interruption(state, effective)
        return ButtonRequestParameters(
            tuple(effective),
            instruction,
            used_pending,
            interruption,
        )

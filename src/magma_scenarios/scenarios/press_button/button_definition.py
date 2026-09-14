# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
from pathlib import Path

from magma_core.simulation.tasks import TaskDefinition
from magma_core.simulation.state import TaskState
from magma_core.simulation.data_structures import SituationInit

from .button_constraints import (
    BUTTON_PRECEDENCE_KEY,
    BUTTON_RULE_SPECS_KEY,
    BUTTON_SEQUENCE_PREFIX_KEY,
)
from .tool import Tool
from .helper import attributes
from .button_request import (
    AskButtonsInExactOrderRequest,
    AskButtonsRequest,
    ForgetButtonRulesRequest,
    GiveButtonGroupOrderRequest,
    GiveEvenOddOrderRequest,
    GiveSequencePrefixRequest,
)
from .rule_renderer import PressButtonRuleRenderer


class PressButtonDefinition(TaskDefinition):
    maniskill_env_id = "PressButtonBasic-v1"
    Tools_cls = Tool
    RuleRenderer_cls = PressButtonRuleRenderer
    active_requests = [
        AskButtonsRequest(),
        AskButtonsInExactOrderRequest(),
        GiveButtonGroupOrderRequest(1),
        GiveEvenOddOrderRequest(),
        GiveSequencePrefixRequest(),
        ForgetButtonRulesRequest(),
    ]

    def __init__(self):
        super().__init__(
            name = "Pressing button definition",
            randomized_config_path=str(Path(__file__).parent.joinpath("press_button_cfg.yaml")),
            situation_init= SituationInit(attributes)
        )
        self.starting_state = TaskState()
        self.starting_state.attributes = attributes
        self.starting_state.relations.update({
            BUTTON_PRECEDENCE_KEY: {},
            BUTTON_SEQUENCE_PREFIX_KEY: {},
        })
        self.starting_state.properties[BUTTON_RULE_SPECS_KEY] = []

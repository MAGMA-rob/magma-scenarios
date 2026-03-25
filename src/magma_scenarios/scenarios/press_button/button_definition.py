# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

# Arthur TANNEAU
from pathlib import Path

from magma_core.base.tasks import TaskDefinition
from magma_core.base.state import TaskState

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


class PressButtonDefinition(TaskDefinition):
    env_id = "PressButtonBasic-v1"
    Tools_cls = Tool
    active_requests = [
        AskButtonsRequest(),
        AskButtonsInExactOrderRequest(),
        GiveButtonGroupOrderRequest(),
        GiveEvenOddOrderRequest(),
        GiveSequencePrefixRequest(),
        ForgetButtonRulesRequest(),
    ]

    def __init__(self):
        super().__init__(
            name = "Pressing button definition",
            randomized_config_path=str(Path(__file__).parent.joinpath("press_button_cfg.yaml"))
        )
        self.starting_state = TaskState()
        self.starting_state.attributes = attributes
        self.starting_state.relations.update({
            BUTTON_PRECEDENCE_KEY: {},
            BUTTON_SEQUENCE_PREFIX_KEY: {},
        })
        self.starting_state.properties[BUTTON_RULE_SPECS_KEY] = []

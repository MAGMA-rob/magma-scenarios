# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.state import TaskState
from magma_scenarios.scenarios.color_sorting.color_requests import AskForCycle
from magma_scenarios.scenarios.warehouse_sorting.warehouse_requests import (
    AddAreas,
    CycleWithPermanentRulesRequest,
    CycleRequest,
    RemoveAreas,
)


def test_add_areas_can_sample_the_last_missing_area():
    state = TaskState()
    state.attributes = {"target_areas": ["area1"]}
    request = AddAreas(["area1", "area2"], max_update=2)

    assert request.sampling_weight(state) > 0

    stages = request.create_stages(state)

    assert len(stages) == 1
    assert stages[0].val_name == "area2"


def test_remove_areas_is_not_sampled_when_only_one_area_remains():
    state = TaskState()
    state.attributes = {"target_areas": ["area1"]}

    assert RemoveAreas().sampling_weight(state) == 0


def test_remove_areas_is_sampled_when_more_than_one_area_remains():
    state = TaskState()
    state.attributes = {"target_areas": ["area1", "area2"]}

    assert RemoveAreas().sampling_weight(state) > 0


def test_cycle_missing_assignment_instruction_has_no_trailing_comma():
    state = TaskState()
    state.attributes = {
        "objects": ["bottle"],
        "target_areas": ["area1"],
    }
    request = CycleRequest()

    stages = request._create_stages(["bottle"], ["area1"], state)

    assert stages[-1].situation.instruction.get_content() == "For this cycle, bottle goes to area1."


def test_permanent_rule_instruction_separates_multiple_assignments():
    state = TaskState()
    state.attributes = {
        "objects": ["bottle", "tray"],
        "target_areas": ["area1", "area2"],
    }
    request = CycleWithPermanentRulesRequest()

    stages = request._create_stages(
        ["bottle", "tray"],
        ["area1", "area2"],
        state,
        base_assignement={"bottle": "area1", "tray": "area2"},
    )

    assert stages[0].situation.instruction.get_content() == (
        "Launch a cycle for bottle and tray. "
        "And consider bottle to area1, tray to area2 as new default assignments."
    )


def test_color_instruction_omits_zero_count_colors():
    request = AskForCycle()

    instruction = request._build_instruction(
        ["green", "black"],
        {"green": 0, "black": 1},
    )

    assert instruction.get_content() == "Please store 1 black cube."

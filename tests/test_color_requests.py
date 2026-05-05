# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.state import TaskState
from magma_scenarios.scenarios.color_sorting.color_requests import (
    AskForCycle,
    FirstColorConstraint,
    GiveOrderConstraint,
)


def _color_state() -> TaskState:
    state = TaskState()
    state.attributes = {"known_box_color": ["green", "yellow"]}
    return state


def test_color_constraint_is_preferred_before_any_cycle_rule_exists():
    state = _color_state()

    assert GiveOrderConstraint().sampling_weight(state) > AskForCycle().sampling_weight(state)


def test_cycle_is_strongly_preferred_until_new_constraint_is_applied():
    state = _color_state()
    FirstColorConstraint().apply(state)

    assert state.properties["constraint_order"] == "first-color"
    assert state.properties["constraint_order_needs_application"] is True
    assert AskForCycle().sampling_weight(state) > GiveOrderConstraint().sampling_weight(state)


def test_cycle_application_marks_current_color_constraint_as_used():
    state = _color_state()
    FirstColorConstraint().apply(state)

    AskForCycle().apply_request(state)

    assert state.properties["constraint_order_needs_application"] is False
    assert state.properties["constraint_order_applications"] == 1


def test_constrained_color_counts_include_both_colors_when_possible():
    request = AskForCycle(max_cube=3)

    for _ in range(20):
        counts = request._sample_color_counts(["green", "yellow"], include_all_colors=True)

        assert counts["green"] > 0
        assert counts["yellow"] > 0


def test_first_color_cycle_samples_a_task_that_exercises_the_rule():
    state = _color_state()
    FirstColorConstraint().apply(state)

    stages = AskForCycle(max_cube=3).create_stages(state)

    final_description = stages[-1].stage_goal_description
    assert "green cube in green box" in final_description
    assert "yellow cube in yellow box" in final_description

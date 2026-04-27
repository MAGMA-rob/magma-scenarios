# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.state import TaskState
from magma_scenarios.scenarios.laundry.laundry_constraints import CLOTHE_DETERGENT_KEY
from magma_scenarios.scenarios.laundry.laundry_request import (
    AskDirectLaundryRequest,
    AskLaundryRequest,
    AssignClotheDetergentRequest,
)
from magma_scenarios.scenarios.make_coffee.coffee_request import (
    AskCoffeePerUser,
    AskCoffeePreferenceInTeam,
    AskCoffeeRequest,
    GiveCoffeePreference,
    GiveTeamCoffeePreference,
)
from magma_scenarios.scenarios.warehouse_sorting.warehouse_requests import CycleRequest
from magma_scenarios.templates.requests import GiveObjectAssignmentRequest


def test_warehouse_assignment_boosts_followup_cycle_until_applied():
    state = TaskState()
    state.attributes = {
        "objects": ["bottle", "tray"],
        "target_areas": ["area1", "area2"],
    }
    assignment_request = GiveObjectAssignmentRequest(max_simultaneous_change=1)

    assignment_request.create_stages(state)
    assignment_request.apply_request(state)

    assert state.properties["object_area_needs_application"] is True
    assert CycleRequest().sampling_weight(state) > assignment_request.sampling_weight(state)

    CycleRequest().apply_request(state)

    assert state.properties["object_area_needs_application"] is False
    assert state.properties["object_area_applications"] == 1


def test_laundry_assignment_boosts_rule_based_wash_over_direct_wash():
    state = TaskState()
    state.attributes = {
        "clothes": ["shirt", "jeans"],
        "detergents": ["detergent_A", "detergent_B"],
    }
    state.relations = {CLOTHE_DETERGENT_KEY: {}}
    assignment_request = AssignClotheDetergentRequest(max_clothes_assignment=1)

    assignment_request.create_stages(state)
    assignment_request.apply_request(state)

    assert state.properties[f"{CLOTHE_DETERGENT_KEY}_needs_application"] is True
    assert AskLaundryRequest().sampling_weight(state) > assignment_request.sampling_weight(state)
    assert AskLaundryRequest().sampling_weight(state) > AskDirectLaundryRequest().sampling_weight(state)

    AskLaundryRequest().apply_request(state)

    assert state.properties[f"{CLOTHE_DETERGENT_KEY}_needs_application"] is False
    assert state.properties[f"{CLOTHE_DETERGENT_KEY}_applications"] == 1


def test_coffee_preference_boosts_user_request_over_direct_coffee():
    state = TaskState()
    state.attributes = {"coffee_pod": ["Latte", "Espresso"]}
    state.relations = {
        "coffee_preference": {},
        "team_coffee_preference_rules": {},
    }
    preference_request = GiveCoffeePreference(["Alice"], max_name=1)

    preference_request.create_stages(state)
    preference_request.apply_request(state)

    assert state.properties["coffee_preference_needs_application"] is True
    assert AskCoffeePerUser(["Alice"]).sampling_weight(state) > preference_request.sampling_weight(state)
    assert AskCoffeePerUser(["Alice"]).sampling_weight(state) > AskCoffeeRequest().sampling_weight(state)

    AskCoffeePerUser(["Alice"]).apply_request(state)

    assert state.properties["coffee_preference_needs_application"] is False
    assert state.properties["coffee_preference_applications"] == 1


def test_team_coffee_preference_boosts_team_preference_question():
    state = TaskState()
    state.attributes = {"coffee_pod": ["Latte", "Espresso"]}
    state.relations = {
        "coffee_preference": {},
        "team_coffee_preference_rules": {},
    }
    teams = {"RND": ["Alice", "Bob"]}
    team_request = GiveTeamCoffeePreference(teams, mode="override")

    team_request.create_stages(state)
    team_request.apply_request(state)

    assert state.properties["team_coffee_preference_needs_application"] is True
    assert AskCoffeePreferenceInTeam(teams).sampling_weight(state) > team_request.sampling_weight(state)

    AskCoffeePreferenceInTeam(teams).apply_request(state)

    assert state.properties["team_coffee_preference_needs_application"] is False
    assert state.properties["team_coffee_preference_applications"] == 1

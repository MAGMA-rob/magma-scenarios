# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import random

from magma_core.base.state import TaskState
from magma_scenarios.scenarios.color_sorting.color_requests import AskForCycle
from magma_scenarios.scenarios.warehouse_sorting.warehouse_definitions import (
    SortingWithInterdictionsDefinition,
)
from magma_scenarios.scenarios.warehouse_sorting.warehouse_requests import (
    AddAreas,
    CycleWithPermanentRulesRequest,
    CycleRequest,
    ForbidObjectsRequest,
    RemoveAreas,
    TemporaryObjectAssignmentCycleRequest,
)
from magma_scenarios.templates.requests import (
    GiveCategoryAssignmentRequest,
    GiveObjectAssignmentRequest,
    GiveObjectCategoryRequest,
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


def test_object_assignment_can_assign_all_objects_to_one_area():
    random.seed(0)
    state = TaskState()
    state.attributes = {
        "objects": ["bottle", "tray", "box"],
        "target_areas": ["area1", "area2"],
    }
    request = GiveObjectAssignmentRequest(assignment_modes=("all_to_one",))

    stages = request.create_stages(state)
    request.apply_request(state)

    assigned_areas = set(state.relations["object_area"].values())
    assert len(request.constraints) == 3
    assert len(assigned_areas) == 1
    assert "all objects" in stages[0].situation.instruction.get_content()


def test_object_assignment_can_split_one_object_from_the_rest():
    random.seed(1)
    state = TaskState()
    state.attributes = {
        "objects": ["bottle", "tray", "box"],
        "target_areas": ["area1", "area2"],
    }
    request = GiveObjectAssignmentRequest(assignment_modes=("split_one_vs_rest",))

    stages = request.create_stages(state)
    request.apply_request(state)

    area_counts = {}
    for area in state.relations["object_area"].values():
        area_counts[area] = area_counts.get(area, 0) + 1

    assert sorted(area_counts.values()) == [1, 2]
    assert "every other object" in stages[0].situation.instruction.get_content()


def test_object_assignment_updates_existing_rule_without_noop():
    random.seed(2)
    state = TaskState()
    state.attributes = {
        "objects": ["bottle"],
        "target_areas": ["area1", "area2"],
    }
    state.relations["object_area"] = {"bottle": "area1"}
    request = GiveObjectAssignmentRequest(assignment_modes=("sample",))

    request.create_stages(state)
    request.apply_request(state)

    assert state.relations["object_area"]["bottle"] == "area2"


def test_object_category_request_reuses_bulk_assignment_template():
    random.seed(3)
    state = TaskState()
    state.attributes = {
        "objects": ["bottle", "tray"],
        "target_areas": ["area1"],
    }
    request = GiveObjectCategoryRequest(
        ["Fragile", "Product A"],
        assignment_modes=("all_to_one",),
    )

    stages = request.create_stages(state)
    request.apply_request(state)

    assert set(state.relations["object_type"].values()) in (
        {"Fragile"},
        {"Product A"},
    )
    assert "all objects are" in stages[0].situation.instruction.get_content()


def test_category_assignment_uses_known_object_categories_as_sources():
    random.seed(4)
    state = TaskState()
    state.attributes = {
        "objects": ["bottle", "tray"],
        "target_areas": ["area1", "area2"],
    }
    state.relations["object_type"] = {
        "bottle": "Fragile",
        "tray": "Product A",
    }
    request = GiveCategoryAssignmentRequest(
        ["Fragile", "Product A", "Unused"],
        assignment_modes=("all_to_one",),
    )

    stages = request.create_stages(state)
    request.apply_request(state)

    assert set(state.relations["type_area"].keys()) == {"Fragile", "Product A"}
    assert "all categories" in stages[0].situation.instruction.get_content()


def test_forbid_objects_request_adds_persistent_interdiction():
    random.seed(5)
    state = TaskState()
    state.attributes = {
        "objects": ["bottle", "tray", "box"],
        "target_areas": ["area1"],
    }
    request = ForbidObjectsRequest()

    request.create_stages(state)
    request.apply_request(state)

    assert len(state.properties["forbidden_objects"]) == 1
    assert state.properties["forbidden_objects"][0] in state.attributes["objects"]


def test_forbid_objects_request_resampling_allows_current_forbidden_object():
    random.seed(6)
    state = TaskState()
    state.attributes = {
        "objects": ["bottle", "tray", "box"],
        "target_areas": ["area1"],
    }
    state.properties["forbidden_objects"] = ["tray"]
    request = ForbidObjectsRequest()

    stages = request.create_stages(state)
    request.apply_request(state)

    assert state.properties["forbidden_objects"] == []
    assert "tray" in stages[0].situation.instruction.get_content()
    assert "again" in stages[0].situation.instruction.get_content()


def test_cycle_request_refuses_forbidden_object_without_cycle_stage():
    state = TaskState()
    state.attributes = {
        "objects": ["bottle", "tray"],
        "target_areas": ["area1"],
    }
    state.properties["forbidden_objects"] = ["tray"]

    stages = CycleRequest()._create_stages(["tray"], ["area1"], state)

    assert len(stages) == 1
    assert "tray is forbidden" in stages[0].verification_prompt


def test_temporary_assignment_cycle_does_not_change_default_rules():
    random.seed(7)
    state = TaskState()
    state.attributes = {
        "objects": ["bottle", "tray", "box"],
        "target_areas": ["area1", "area2"],
    }
    state.relations["object_area"] = {"bottle": "area1"}
    request = TemporaryObjectAssignmentCycleRequest(all_objects_probability=1)

    stages = request.create_stages(state)
    request.apply_request(state)

    assert set(stages[0].situation.attributes["objects"]) == {"bottle", "tray", "box"}
    assert state.relations["object_area"] == {"bottle": "area1"}


def test_temporary_assignment_cycle_refuses_forbidden_object():
    random.seed(8)
    state = TaskState()
    state.attributes = {
        "objects": ["bottle", "tray"],
        "target_areas": ["area1"],
    }
    state.properties["forbidden_objects"] = ["tray"]
    request = TemporaryObjectAssignmentCycleRequest(all_objects_probability=1)

    stages = request.create_stages(state)

    assert len(stages) == 1
    assert "tray is forbidden" in stages[0].verification_prompt


def test_sorting_with_interdictions_definition_uses_new_requests():
    request_names = [
        request.__class__.__name__
        for request in SortingWithInterdictionsDefinition.active_requests
    ]

    assert "ForbidObjectsRequest" in request_names
    assert "TemporaryObjectAssignmentCycleRequest" in request_names

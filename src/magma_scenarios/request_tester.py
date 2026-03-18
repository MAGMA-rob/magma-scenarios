# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

"""
Utility to inspect how a TaskDefinition reacts to a chosen request and state.

Example usages:

python -m magma_scenarios.request_tester warehouse_sorting.SimpleSortingDefinition

python -m magma_scenarios.request_tester \
    warehouse_sorting.SortingCategoryDefinition \
    --setup-request GiveObjectCategoryRequest \
    --setup-request GiveCategoryAssignmentRequest \
    --request CycleByCategoriesRequest \
    --seed 0

python -m magma_scenarios.request_tester \
    warehouse_sorting.SimpleSortingDefinition \
    --request RemoveAreas \
    --state-file /tmp/custom_state.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from magma_core.base.state import TaskState
from .registry_loader import load_definition


STATE_OVERRIDE_KEYS = {
    "memory",
    "preserved_memory_indices",
    "attributes",
    "relations",
    "properties",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a TaskDefinition request from a chosen TaskState. "
            "You can list requests, prepare the state with setup requests, "
            "override the starting state from JSON, then run one specific request."
        )
    )
    parser.add_argument(
        "definition",
        nargs="?",
        type=str,
        help=(
            "TaskDefinition registry name "
            "(e.g. warehouse_sorting.SimpleSortingDefinition)."
        ),
    )
    parser.add_argument(
        "--request",
        type=str,
        default=None,
        help=(
            "Request to execute after the optional setup requests. "
            "Can be a class name or an index from the available requests list."
        ),
    )
    parser.add_argument(
        "--setup-request",
        action="append",
        default=[],
        help=(
            "Request to apply before the target request. Can be repeated. "
            "Useful to populate relations before testing another request."
        ),
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file that overrides the starting state. "
            "Supported keys: memory, preserved_memory_indices, attributes, relations, properties."
        ),
    )
    parser.add_argument(
        "--state-json",
        type=str,
        default=None,
        help=(
            "Inline JSON override for the starting state. Same schema as --state-file. "
            "Applied after --state-file if both are provided."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for Python's random module.",
    )
    parser.add_argument(
        "--print-state-template",
        action="store_true",
        help="Print an example JSON state override and exit.",
    )
    return parser.parse_args()
def load_state_override(args: argparse.Namespace) -> Dict[str, Any]:
    data: Dict[str, Any] = {}

    if args.state_file is not None:
        if not args.state_file.exists():
            raise FileNotFoundError(f"Impossible to find state file: {args.state_file}")
        with args.state_file.open("r", encoding="utf-8") as file:
            data.update(json.load(file))

    if args.state_json is not None:
        data.update(json.loads(args.state_json))

    unknown_keys = sorted(set(data.keys()) - STATE_OVERRIDE_KEYS)
    if unknown_keys:
        raise ValueError(
            "Unsupported state override keys: "
            + ", ".join(unknown_keys)
            + ". Supported keys are: "
            + ", ".join(sorted(STATE_OVERRIDE_KEYS))
        )

    return data


def apply_state_override(base_state: TaskState, data: Dict[str, Any]) -> TaskState:
    state = base_state.clone()
    for key, value in data.items():
        setattr(state, key, copy.deepcopy(value))
    return state


def get_request_catalog(definition) -> List[Any]:
    return list(definition.active_requests)


def resolve_request(requests: List[Any], selector: str) -> Tuple[int, Any]:
    if selector.isdigit():
        index = int(selector)
        if index < 0 or index >= len(requests):
            raise IndexError(f"Request index {index} is out of bounds.")
        return index, requests[index]

    normalized = selector.strip().lower()
    candidates = []
    for index, request in enumerate(requests):
        class_name = request.__class__.__name__
        qualified_name = f"{request.__class__.__module__}.{class_name}"
        if (
            class_name.lower() == normalized
            or qualified_name.lower() == normalized
            or normalized in class_name.lower()
        ):
            candidates.append((index, request))

    if not candidates:
        raise ValueError(f"Unable to find a request matching '{selector}'.")
    if len(candidates) > 1:
        matches = ", ".join(f"[{idx}] {req.__class__.__name__}" for idx, req in candidates)
        raise ValueError(f"Ambiguous request selector '{selector}'. Matches: {matches}")

    return candidates[0]


def print_state(title: str, state: TaskState) -> None:
    print(f"\n=== {title} ===")
    print(state.to_human_readable())


def print_request_list(definition, state: TaskState) -> None:
    print("\n=== Available requests ===")
    for index, request in enumerate(get_request_catalog(definition)):
        try:
            weight = request.sampling_weight(state.clone())
        except Exception as exc:
            weight = f"error: {exc}"
        print(f"[{index}] {request.__class__.__name__} | sampling_weight={weight}")


def format_goal_names(goals: Iterable[Any]) -> str:
    goal_names = [goal.__class__.__name__ for goal in goals]
    return ", ".join(goal_names) if goal_names else "none"


def print_stages(stages: List[Any]) -> None:
    print(f"\n=== Generated stages ({len(stages)}) ===")
    if not stages:
        print("No stage generated.")
        return

    for index, stage in enumerate(stages):
        situation = stage.situation
        instruction = situation.instruction.get_content()
        print(f"\n[{index}] {stage.__class__.__name__}")
        print(f"  goal_description: {stage.stage_goal_description}")
        print(f"  instruction: {instruction if instruction else '<EMPTY>'}")
        print(f"  reset_at_end: {stage.reset_at_end}")
        print(f"  additive_stage: {getattr(stage, 'additive_stage', False)}")
        print(f"  flag_answer_to_user: {situation.flag_answer_to_user}")
        print(f"  goals: {format_goal_names(stage.goals)}")
        if hasattr(stage, "verification_prompt"):
            print(f"  verification_prompt: {stage.verification_prompt}")
        # print(f"  memory: {situation.memory}")
        # print(f"  preserved_memory_indices: {situation.preserved_memory_indices}")
        # print(f"  attributes: {json.dumps(situation.attributes, ensure_ascii=True, indent=2)}")


def execute_request(state: TaskState, request: Any, base_state: TaskState) -> Tuple[List[Any], TaskState]:
    sampled_request = copy.deepcopy(request)
    stages = sampled_request.create_stages(state)
    next_state = sampled_request.apply_request(state)
    if sampled_request.force_state_recompute():
        next_state = next_state.recompute_from_base(base_state)
    return stages, next_state


def run_request(
    label: str,
    state: TaskState,
    request: Any,
    request_index: int,
    base_state: TaskState,
) -> TaskState:
    print(f"\n=== {label} ===")
    print(f"Request: [{request_index}] {request.__class__.__name__}")
    print(f"Sampling weight on current state: {request.sampling_weight(state.clone())}")

    stages, next_state = execute_request(state, request, base_state)
    print_stages(stages)
    print_state(f"State after {request.__class__.__name__}", next_state)
    return next_state


def print_state_template() -> None:
    template = {
        "memory": ["Optional memory line"],
        "preserved_memory_indices": [0],
        "attributes": {
            "objects": ["obj_1", "obj_2"],
            "target_areas": ["zone_a", "zone_b"],
        },
        "relations": {
            "object_type": {"obj_1": "fragile"},
            "type_area": {"fragile": "zone_a"},
            "object_area": {"obj_2": "zone_b"},
        },
        "properties": {
            "forbidden_objects": [],
            "forbidden_areas": [],
        },
    }
    print(json.dumps(template, indent=2, ensure_ascii=True))


def main() -> int:
    args = parse_args()

    if args.print_state_template:
        print_state_template()
        return 0

    if args.definition is None:
        raise ValueError("A TaskDefinition reference is required unless you use --print-state-template.")

    random.seed(args.seed)

    definition_cls = load_definition(args.definition)
    definition = definition_cls()

    state_override = load_state_override(args)
    state = apply_state_override(definition.starting_state, state_override)
    base_state = state.clone()

    print(f"Definition: {definition_cls.__module__}.{definition_cls.__name__}")
    print(f"Seed: {args.seed}")
    print_state("Initial state", state)
    print_request_list(definition, state)

    if args.request is None:
        return 0

    for selector in args.setup_request:
        request_index, request = resolve_request(get_request_catalog(definition), selector)
        state = run_request("Setup request", state, request, request_index, base_state)

    request_index, request = resolve_request(get_request_catalog(definition), args.request)
    run_request("Target request", state, request, request_index, base_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

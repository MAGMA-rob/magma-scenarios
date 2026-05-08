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

python -m magma_scenarios.request_tester \f
    warehouse_sorting.SimpleSortingDefinition \
    --request RemoveAreas \
    --state-file /tmp/custom_state.json

python -m magma_scenarios.request_tester \
    warehouse_sorting.SimpleSortingDefinition \
    --sample 50

python -m magma_scenarios.request_tester \
    warehouse_sorting.SimpleSortingDefinition \
    --sample 3 \
    --detailled
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from magma_core.base.state import TaskState
from .registry_loader import load_definition


STATE_OVERRIDE_KEYS = {
    "memory",
    "preserved_memory_indices",
    "attributes",
    "relations",
    "properties",
}
DEFAULT_MAX_TOTAL_TARGET_STEPS = 15


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
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Generate N random tasks from the definition, restarting from the "
            "original TaskState for each task. Errors are printed with context."
        ),
    )
    parser.add_argument(
        "--detailed",
        "--detailled",
        dest="sample_detailed",
        action="store_true",
        help=(
            "With --sample, print each sampled request, generated stages, "
            "instructions, goals, and resulting attributes/properties state."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sample is not None and args.sample <= 0:
        raise ValueError("--sample must be a positive integer.")
    if args.sample is not None and args.request is not None:
        raise ValueError("--sample generates full random tasks; do not combine it with --request.")
    if args.sample is None and args.sample_detailed:
        raise ValueError("--detailed/--detailled must be used with --sample.")


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
        for idx, req in candidates:
            if req.__class__.__name__ == selector:
                return idx, req
        matches = ", ".join(f"[{idx}] {req.__class__.__name__}" for idx, req in candidates)
        raise ValueError(f"Ambiguous request selector '{selector}'. Matches: {matches}")

    return candidates[0]


def print_state(title: str, state: TaskState) -> None:
    print(f"\n=== {title} ===")
    print(state.to_human_readable())


def print_compact_state(title: str, state: TaskState) -> None:
    print(f"\n=== {title} ===")
    state_data = {
        "attributes": state.attributes,
        "properties": state.properties,
    }
    print(json.dumps(state_data, ensure_ascii=True, indent=2, default=repr))


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
        print(f"  target_steps: {stage.target_steps}")
        print(f"  reset_at_end: {stage.reset_at_end}")
        print(f"  additive_stage: {getattr(stage, 'additive_stage', False)}")
        print(f"  flag_answer_to_user: {situation.flag_answer_to_user}")
        print(f"  goals: {format_goal_names(stage.goals)}")
        if hasattr(stage, "verification_prompt"):
            print(f"  verification_prompt: {stage.verification_prompt}")
        # print(f"  memory: {situation.memory}")
        # print(f"  preserved_memory_indices: {situation.preserved_memory_indices}")
        # print(f"  attributes: {json.dumps(situation.attributes, ensure_ascii=True, indent=2)}")


def print_sample_stages(stages: List[Any]) -> None:
    print(f"\nGenerated stages: {len(stages)}")
    if not stages:
        print("No stage generated.")
        return

    for index, stage in enumerate(stages):
        situation = stage.situation
        instruction = situation.instruction.get_content()
        print(f"\n[{index}] {stage.__class__.__name__}")
        print(f"  instruction: {instruction if instruction else '<EMPTY>'}")
        print(f"  goal_description: {stage.stage_goal_description}")
        print(f"  goals: {format_goal_names(stage.goals)}")
        print(f"  flag_answer_to_user: {situation.flag_answer_to_user}")
        print(f"  reset_at_end: {stage.reset_at_end}")


def get_weight(request: Any, state: TaskState) -> float:
    weight = request.sampling_weight(state.clone())
    if weight < 0:
        raise ValueError(
            f"{request.__class__.__name__}.sampling_weight returned a negative weight: {weight}"
        )
    return weight


def select_weighted_request(
    definition: Any,
    state: TaskState,
) -> Tuple[Optional[int], Optional[Any], List[Dict[str, Any]]]:
    requests = get_request_catalog(definition)
    eligible: List[Tuple[int, Any]] = []
    weights: List[float] = []
    errors: List[Dict[str, Any]] = []

    for request_index, request in enumerate(requests):
        try:
            weight = get_weight(request, state)
        except Exception as exc:
            errors.append(
                {
                    "request_index": request_index,
                    "request_name": request.__class__.__name__,
                    "phase": "sampling_weight",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            continue

        if weight > 0:
            eligible.append((request_index, request))
            weights.append(weight)

    if not eligible:
        return None, None, errors

    selected_position = random.choices(range(len(eligible)), weights=weights, k=1)[0]
    request_index, request = eligible[selected_position]
    return request_index, request, errors


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


def print_task_error(
    sample_error: Dict[str, Any],
) -> None:
    print(
        f"\nTask [{sample_error['task_index']}] ERROR "
        f"seed={sample_error['task_seed']} "
        f"phase={sample_error['phase']} "
        f"request={sample_error['request_label']}"
    )
    print(f"  {sample_error['error']}")

    if sample_error["request_history"]:
        print("  Requests already sampled in this task:")
        for request_label in sample_error["request_history"]:
            print(f"    - {request_label}")

    print("  State before error:")
    for line in sample_error["state"].splitlines():
        print(f"    {line}")

    if sample_error.get("traceback"):
        print("  Traceback:")
        for line in sample_error["traceback"].rstrip().splitlines():
            print(f"    {line}")


def run_sampling(
    definition: Any,
    initial_state: TaskState,
    sample_count: int,
    seed: int,
    detailed: bool = False,
    max_total_target_steps: int = DEFAULT_MAX_TOTAL_TARGET_STEPS,
) -> None:
    errors: List[Dict[str, Any]] = []
    request_counts: Counter[str] = Counter()
    successes = 0

    print("\n=== Random task sampling ===")
    print(f"Tasks to generate: {sample_count}")
    print(f"State handling: restart from the same initial TaskState for every task")
    print(f"Max total target steps per task: {max_total_target_steps}")
    if detailed:
        print("Detailed logging: enabled")

    for task_index in range(sample_count):
        task_seed = seed + task_index
        random.seed(task_seed)

        base_state = initial_state.clone()
        state = base_state.clone()
        task = definition.build_default_task()
        task.stages = []
        total_target_steps = 0
        request_history: List[str] = []
        failed = False

        if detailed:
            print(f"\n=== Task [{task_index}] detail seed={task_seed} ===")

        while total_target_steps < max_total_target_steps:
            request_index, request, selection_errors = select_weighted_request(definition, state)

            if selection_errors:
                selection_error = selection_errors[0]
                request_label = (
                    f"[{selection_error['request_index']}] {selection_error['request_name']}"
                )
                sample_error = {
                    "task_index": task_index,
                    "task_seed": task_seed,
                    "request_label": request_label,
                    "phase": selection_error["phase"],
                    "error": selection_error["error"],
                    "request_history": request_history,
                    "state": state.to_human_readable(),
                    "traceback": selection_error["traceback"],
                }
                errors.append(sample_error)
                print_task_error(sample_error)
                failed = True
                break

            if request is None:
                if detailed:
                    print("No eligible request remains for this state.")
                break

            request_label = f"[{request_index}] {request.__class__.__name__}"
            state_before_request = state.clone()
            try:
                stages, next_state = execute_request(state, request, base_state)
                sampled_target_steps = sum(stage.target_steps for stage in stages)
            except Exception as exc:
                sample_error = {
                    "task_index": task_index,
                    "task_seed": task_seed,
                    "request_label": request_label,
                    "phase": "create/apply",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "request_history": request_history,
                    "state": state_before_request.to_human_readable(),
                    "traceback": traceback.format_exc(),
                }
                errors.append(sample_error)
                print_task_error(sample_error)
                failed = True
                break

            if task.stages and total_target_steps + sampled_target_steps > max_total_target_steps:
                if detailed:
                    print(
                        f"\nStopping before {request_label}: "
                        f"{total_target_steps} + {sampled_target_steps} target steps "
                        f"would exceed max {max_total_target_steps}."
                    )
                break

            if detailed:
                print(f"\n--- Sampled request {request_label} ---")
                print_sample_stages(stages)
                print_compact_state(f"State after {request.__class__.__name__}", next_state)

            task.stages.extend(stages)
            total_target_steps += sampled_target_steps
            state = next_state
            request_counts[request.__class__.__name__] += 1
            request_history.append(
                f"{request_label} -> {len(stages)} stage(s), {sampled_target_steps} target step(s)"
            )

        if failed:
            print(f"Task [{task_index}] FAILED seed={task_seed}")
            continue

        successes += 1
        print(
            f"Task [{task_index}] OK seed={task_seed} "
            f"stages={len(task.stages)} target_steps={total_target_steps} "
            f"requests={len(request_history)}"
        )

    print("\n=== Sampling summary ===")
    print(f"Successful tasks: {successes}/{sample_count}")
    print(f"Errors: {len(errors)}")

    total_requests = sum(request_counts.values())
    print(f"Sampled requests: {total_requests}")
    if request_counts:
        print("\nRequest type counts:")
        for request_name, count in sorted(
            request_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            percentage = count / total_requests * 100
            print(f"- {request_name}: {count} ({percentage:.1f}%)")

    if not errors:
        return

    grouped: Dict[Tuple[str, str, str], int] = {}
    for sample_error in errors:
        key = (
            sample_error["request_label"],
            sample_error["phase"],
            sample_error["error"],
        )
        grouped[key] = grouped.get(key, 0) + 1

    print("\nGrouped errors:")
    for (request_label, phase, error), count in sorted(
        grouped.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2]),
    ):
        print(f"- {count}x {request_label} | {phase} | {error}")


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
    validate_args(args)

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
    if args.sample_detailed:
        print_compact_state("Initial state", state)
    else:
        print_state("Initial state", state)
    print_request_list(definition, state)

    for selector in args.setup_request:
        request_index, request = resolve_request(get_request_catalog(definition), selector)
        state = run_request("Setup request", state, request, request_index, base_state)

    if args.sample is not None:
        run_sampling(
            definition=definition,
            initial_state=state,
            sample_count=args.sample,
            seed=args.seed,
            detailed=args.sample_detailed,
        )
        return 0

    if args.request is None:
        return 0

    request_index, request = resolve_request(get_request_catalog(definition), args.request)
    run_request("Target request", state, request, request_index, base_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

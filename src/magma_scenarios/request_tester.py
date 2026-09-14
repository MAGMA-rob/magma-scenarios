# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

"""Generate and inspect random request chains for a task definition.

This module deliberately has only two uses:

* ``mass`` generates many tasks quietly.  Use it as a regression check and to
  see which requests and stage classes are actually reached.
* ``trace`` generates a few tasks verbosely.  Before every draw it prints all
  request weights, then shows the selected request and the stages it produced.

``--setup`` is available in both modes.  A setup request is applied once before
generation, which makes it possible to test chains that need a prepared state.

Examples::

    python -m magma_scenarios.request_tester \
        warehouse_sorting.SimpleSortingDefinition mass --count 1000

    python -m magma_scenarios.request_tester \
        warehouse_sorting.SortingCategoryDefinition trace --count 3 \
        --setup GiveObjectCategoryRequest --setup GiveCategoryAssignmentRequest
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magma_core.simulation.state import TaskState

from magma_scenarios import load_definition


DEFAULT_COUNT = 100
DEFAULT_MAX_STAGES = 15
# A request can legitimately create no stage.  This independent guard keeps a
# faulty request/state pair from making the tester loop forever.
MAX_REQUESTS_PER_TASK = 100


@dataclass
class GenerationReport:
    task_count: int
    successful_tasks: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    request_counts: Counter = field(default_factory=Counter)
    stage_counts: Counter = field(default_factory=Counter)
    stages_by_request: DefaultDict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random request chains from a TaskDefinition.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  mass   Generate many tasks and print a compact coverage/error summary.\n"
            "  trace  Generate a few tasks and print weights, selected requests and stages.\n\n"
            "A setup request is applied once before every generated task. It is useful "
            "when the requests under test require a state prepared by earlier requests."
        ),
    )
    parser.add_argument(
        "definition",
        help="Registered TaskDefinition (for example warehouse_sorting.SimpleSortingDefinition).",
    )
    parser.add_argument("mode", choices=("mass", "trace"), help="Generation mode.")
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of tasks to generate (default: {DEFAULT_COUNT}).",
    )
    parser.add_argument(
        "--setup",
        action="append",
        default=[],
        metavar="REQUEST",
        help="Request to apply before generation; may be repeated.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed of the first task (default: 0).")
    parser.add_argument(
        "--max-stages",
        type=int,
        default=DEFAULT_MAX_STAGES,
        metavar="N",
        help=(
            "Maximum number of stages in one generated task "
            f"(default: {DEFAULT_MAX_STAGES})."
        ),
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be a positive integer.")
    if args.max_stages <= 0:
        parser.error("--max-stages must be a positive integer.")
    return args


def request_catalog(definition: Any) -> List[Any]:
    return list(definition.active_requests)


def request_name(request: Any, index: int) -> str:
    return f"[{index}] {request.__class__.__name__}"


def resolve_request(requests: Sequence[Any], selector: str) -> Tuple[int, Any]:
    """Resolve a setup request by index, exact class name or unique substring."""
    if selector.isdigit():
        index = int(selector)
        if 0 <= index < len(requests):
            return index, requests[index]
        raise ValueError(f"Setup request index {index} is out of bounds.")

    normalized = selector.strip().lower()
    matches = [
        (index, request)
        for index, request in enumerate(requests)
        if normalized in request.__class__.__name__.lower()
    ]
    exact_matches = [
        match for match in matches if match[1].__class__.__name__.lower() == normalized
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if not matches:
        raise ValueError(f"Unable to find setup request '{selector}'.")
    if len(matches) > 1:
        choices = ", ".join(request_name(request, index) for index, request in matches)
        raise ValueError(f"Ambiguous setup request '{selector}'. Matches: {choices}")
    return matches[0]


def execute_request(state: TaskState, request: Any, base_state: TaskState) -> Tuple[List[Any], TaskState]:
    """Run a private request instance, following magma_core's request contract."""
    sampled_request = copy.deepcopy(request)
    parameters = sampled_request.sample_parameters(state)
    stages = sampled_request.create_stages(state, parameters)
    next_state = sampled_request.apply_request(state, parameters)
    if sampled_request.force_state_recompute():
        next_state = next_state.recompute_from_base(base_state)
    return list(stages), next_state


def weights_for_state(requests: Sequence[Any], state: TaskState) -> Tuple[List[float], Optional[Dict[str, Any]]]:
    """Return usable weights, or a diagnostic if one request cannot be evaluated."""
    weights: List[float] = []
    for index, request in enumerate(requests):
        try:
            weight = request.sampling_weight(state.clone())
            if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
                raise ValueError(f"sampling_weight must be a finite non-negative number, got {weight!r}")
        except Exception as exc:
            return [], {
                "request": request_name(request, index),
                "phase": "sampling_weight",
                "error": f"{exc.__class__.__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        weights.append(float(weight))
    return weights, None


def print_weights(requests: Sequence[Any], weights: Sequence[float]) -> None:
    print("  Available request weights:")
    for index, (request, weight) in enumerate(zip(requests, weights)):
        print(f"    {request_name(request, index)}: {weight:g}")


def format_goal_names(goals: Iterable[Any]) -> str:
    names = [goal.__class__.__name__ for goal in goals]
    return ", ".join(names) if names else "none"


def print_stages(stages: Sequence[Any]) -> None:
    if not stages:
        print("    Stages: none")
        return
    print(f"    Stages ({len(stages)}):")
    for index, stage in enumerate(stages, start=1):
        situation = stage.get_stage_input()
        instruction = situation.instruction.get_content() or "<EMPTY>"
        print(f"      {index}. {stage.__class__.__name__}")
        print(f"         instruction: {instruction}")
        print(f"         goal: {stage.get_stage_goal_description()}")
        print(f"         goals: {format_goal_names(stage.get_goals())}")
        print(f"         target_tool_calls: {stage.get_target_tool_calls()}")
        print(f"         max_tool_calls: {stage.get_max_tool_calls()}")
        print(f"         linked_to_previous: {stage.is_linked_to_prev()}")
        print(f"         reset_at_end: {stage.should_reset_at_end()}")


def error_record(
    task_index: int,
    task_seed: int,
    phase: str,
    request: str,
    history: Sequence[str],
    state: TaskState,
    exc: Optional[BaseException] = None,
) -> Dict[str, Any]:
    return {
        "task_index": task_index,
        "task_seed": task_seed,
        "phase": phase,
        "request": request,
        "error": "request limit reached" if exc is None else f"{exc.__class__.__name__}: {exc}",
        "history": list(history),
        "state": state.to_human_readable(),
        "traceback": "" if exc is None else traceback.format_exc(),
    }


def print_error(error: Dict[str, Any], detailed: bool) -> None:
    print(
        f"ERROR task={error['task_index']} seed={error['task_seed']} "
        f"phase={error['phase']} request={error['request']}: {error['error']}"
    )
    if detailed:
        if error["history"]:
            print("  Previous requests: " + " -> ".join(error["history"]))
        print("  State before error:")
        for line in error["state"].splitlines():
            print(f"    {line}")
        if error["traceback"]:
            print(error["traceback"].rstrip())


def generate_task(
    definition: Any,
    initial_state: TaskState,
    task_index: int,
    seed: int,
    max_stages: int,
    detailed: bool,
) -> Tuple[List[Tuple[str, List[Any]]], Optional[Dict[str, Any]]]:
    random.seed(seed)
    requests = request_catalog(definition)
    base_state = initial_state.clone()
    state = base_state.clone()
    history: List[str] = []
    generated: List[Tuple[str, List[Any]]] = []
    stage_count = 0

    if detailed:
        print(f"\n=== Trace {task_index + 1} (seed {seed}) ===")

    for _ in range(MAX_REQUESTS_PER_TASK):
        weights, weight_error = weights_for_state(requests, state)
        if weight_error is not None:
            error = error_record(task_index, seed, weight_error["phase"], weight_error["request"], history, state)
            error["error"] = weight_error["error"]
            error["traceback"] = weight_error["traceback"]
            return generated, error
        if detailed:
            print_weights(requests, weights)
        if not any(weights):
            if detailed:
                print("  No eligible request remains; trace complete.")
            return generated, None

        index = random.choices(range(len(requests)), weights=weights, k=1)[0]
        request = requests[index]
        label = request_name(request, index)
        try:
            stages, next_state = execute_request(state, request, base_state)
        except Exception as exc:
            return generated, error_record(task_index, seed, "create_stages/apply_request", label, history, state, exc)

        # A request is atomic: if it would exceed the requested task size, it
        # is not added.  This preserves the state/stage correspondence.
        if stage_count + len(stages) > max_stages:
            if detailed:
                print(f"  Stop before {label}: stage limit ({max_stages}) would be exceeded.")
            return generated, None

        if detailed:
            print(f"  Selected {label}")
            print_stages(stages)
        generated.append((label, stages))
        history.append(label)
        stage_count += len(stages)
        state = next_state
        if stage_count >= max_stages:
            if detailed:
                print(f"  Stage limit reached ({max_stages}); trace complete.")
            return generated, None

    return generated, error_record(task_index, seed, "generation", "<none>", history, state)


def apply_setup(definition: Any, setup_selectors: Sequence[str]) -> TaskState:
    state = definition.starting_state.clone()
    base_state = state.clone()
    requests = request_catalog(definition)
    for selector in setup_selectors:
        index, request = resolve_request(requests, selector)
        label = request_name(request, index)
        print(f"Setup: {label}")
        stages, state = execute_request(state, request, base_state)
        print(f"  Applied ({len(stages)} stage(s) generated and discarded).")
    return state


def run(definition: Any, initial_state: TaskState, args: argparse.Namespace) -> GenerationReport:
    detailed = args.mode == "trace"
    report = GenerationReport(task_count=args.count)
    for task_index in range(args.count):
        generated, error = generate_task(
            definition, initial_state, task_index, args.seed + task_index, args.max_stages, detailed
        )
        if error is not None:
            report.errors.append(error)
            print_error(error, detailed)
            continue
        report.successful_tasks += 1
        for request, stages in generated:
            report.request_counts[request] += 1
            for stage in stages:
                stage_name = stage.__class__.__name__
                report.stage_counts[stage_name] += 1
                report.stages_by_request[request][stage_name] += 1
    return report


def print_summary(report: GenerationReport) -> None:
    print("\n=== Summary ===")
    print(f"Tasks: {report.successful_tasks}/{report.task_count} successful")
    print(f"Errors: {len(report.errors)}")
    print(f"Requests sampled: {sum(report.request_counts.values())}")
    print(f"Stages generated: {sum(report.stage_counts.values())}")
    if report.request_counts:
        print("\nRequests sampled:")
        for name, count in report.request_counts.most_common():
            print(f"  {name}: {count}")
    if report.stage_counts:
        print("\nStage types encountered:")
        for name, count in report.stage_counts.most_common():
            print(f"  {name}: {count}")
        print("\nStages produced by request:")
        for request in sorted(report.stages_by_request):
            stages = ", ".join(
                f"{stage} ({count})" for stage, count in report.stages_by_request[request].most_common()
            )
            print(f"  {request}: {stages or 'none'}")
    if report.errors:
        grouped = Counter((error["phase"], error["request"], error["error"]) for error in report.errors)
        print("\nErrors by cause:")
        for (phase, request, message), count in grouped.most_common():
            print(f"  {count}x {phase} | {request} | {message}")


def main() -> int:
    args = parse_args()
    definition_cls = load_definition(args.definition)
    definition = definition_cls()
    print(f"Definition: {definition_cls.__module__}.{definition_cls.__name__}")
    print(f"Mode: {args.mode}; count: {args.count}; max stages: {args.max_stages}; seed: {args.seed}")
    initial_state = apply_setup(definition, args.setup)
    report = run(definition, initial_state, args)
    print_summary(report)
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

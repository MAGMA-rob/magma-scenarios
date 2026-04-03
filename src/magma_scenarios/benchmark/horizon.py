# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Mapping


LENGTH_GROUPS = {
    "2-3": (2, 3),
    "4-5": (4, 5),
    "6-9": (6, 9),
    "10-16": (10, 16),
}


def _resolve_expected_behavior(stage: Mapping[str, Any]) -> str:
    expected_behavior = stage.get("expected_behavior", "act")
    if expected_behavior not in {"act", "acknowledge", "answer"}:
        raise ValueError(
            "expected_behavior must be one of 'act', 'acknowledge' or 'answer'. "
            f"Got {expected_behavior!r}."
        )
    return expected_behavior


def _iter_injections(stage: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_injections = stage.get("injection", stage.get("injections", None))
    if raw_injections is None:
        return []
    if isinstance(raw_injections, Mapping):
        return [raw_injections]
    if not isinstance(raw_injections, list):
        raise TypeError("Field 'injection' must be either a dict or a list of dict.")
    for injection in raw_injections:
        if not isinstance(injection, Mapping):
            raise TypeError("Each injection entry must be a mapping.")
    return list(raw_injections)


def stage_has_recovery(stage: Mapping[str, Any]) -> bool:
    if stage.get("force_recovery", None) is not None or stage.get("force_failure", None) is not None:
        raise ValueError(
            "Legacy fields 'force_recovery' and 'force_failure' are not supported anymore. "
            "Please migrate benchmark tasks to the 'injection' field."
        )

    return any(
        injection.get("mode") in {"force_recovery", "force_failure"}
        for injection in _iter_injections(stage)
    )


def count_stage_horizon(stage: Mapping[str, Any]) -> int:
    expected_behavior = _resolve_expected_behavior(stage)
    default_max_step = 1 if expected_behavior in {"acknowledge", "answer"} else None
    max_step = stage.get("max_step", default_max_step)

    if not isinstance(max_step, int) or max_step <= 0:
        raise ValueError(f"Invalid stage horizon contribution from max_step={max_step!r}.")

    horizon = max_step
    if stage.get("answer_to_user", None) or stage.get("flag_answer_to_user", None):
        horizon += 1
    if stage_has_recovery(stage):
        horizon += 1
    return horizon


def count_task_horizon(stage_list: list[Mapping[str, Any]]) -> int:
    return sum(count_stage_horizon(stage) for stage in stage_list)


def task_has_recovery_criterion(stage_list: list[Mapping[str, Any]]) -> bool:
    return any(stage_has_recovery(stage) for stage in stage_list)


def get_length_bucket(task_horizon: int) -> str:
    for bucket_name, (lower, upper) in LENGTH_GROUPS.items():
        if lower <= task_horizon <= upper:
            return bucket_name
    return "other"

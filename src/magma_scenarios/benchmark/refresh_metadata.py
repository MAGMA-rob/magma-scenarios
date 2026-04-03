# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import json
from pathlib import Path
from collections import Counter, defaultdict

from .criteria import KNOWN_CRITERIA
from .horizon import count_task_horizon, task_has_recovery_criterion

def main():

    base = Path(__file__).resolve().parent
    bench_metadata = []

    required_meta_fields = [
            "scenario_name",
            "scenario_id",
            "scenario_description",
            "associate_task",
            "options_mplib"
        ]

    for folder in base.iterdir():
        if not folder.is_dir():
            continue

        tasks_folder = folder / "tasks"
        meta_file = folder / "meta_info.json"

        if not tasks_folder.exists() or not meta_file.exists():
            continue

        # Read and verify meta_info.json
        with meta_file.open("r") as f:
            meta_info = json.load(f)

        for field in required_meta_fields:
            if field not in meta_info:
                raise ValueError(f"Missing '{field}' in {meta_file}")

        # Count criteria from task files
        criteria_counter = Counter()
        criteria_tasks = defaultdict(list)
        tasks_horizon = defaultdict(list)

        for task_file in tasks_folder.iterdir():
            if task_file.suffix != ".json":
                continue
            
            with task_file.open("r") as f:
                task_data = json.load(f)
            
            if "stages" not in task_data or "criteria" not in task_data:
                raise ValueError(f"task file {task_file} missing 'stages' or 'criteria'")
            
            effective_criteria = list(dict.fromkeys(task_data["criteria"]))
            if task_has_recovery_criterion(task_data["stages"]) and "recovery" not in effective_criteria:
                effective_criteria.append("recovery")

            for c in effective_criteria:
                if c not in KNOWN_CRITERIA: 
                    raise ValueError(f"Unknown criteria '{c}' in {task_file}") 
                criteria_counter[c] += 1
                criteria_tasks[c].append(task_file.name)
            
            task_horizon = count_task_horizon(task_data["stages"])
            tasks_horizon[task_horizon].append(task_file.name.split(".")[0])

        # Update meta_info.json with criteria counts
        meta_info["tasks_horizon"] = dict(tasks_horizon)
        meta_info["criteria"] = dict(criteria_counter)
        meta_info["criteria_tasks"] = dict(criteria_tasks)
        with meta_file.open("w") as f:
            json.dump(meta_info, f, indent=2)

        # Prepare entry for bench_metadata.json
        present_criteria = [c for c, count in criteria_counter.items() if count > 0]
        bench_metadata.append({
            "scenario_name": meta_info["scenario_name"],
            "folder_name": folder.name,
            "criteria": present_criteria
        })

    with (base / "bench_metadata.json").open("w") as f:
        json.dump(bench_metadata, f, indent=2)

if __name__ == "__main__":
    main()

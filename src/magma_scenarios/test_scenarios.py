import numpy as np
import argparse
from typing import List, Dict, Type, Tuple
import importlib, pkgutil

from magma_core.simulation.tasks import BaseTask

def collect_task_classes(package_name: str, scenarios: List[str]) -> List[Type[BaseTask]]:
    task_classes = []

    pkg = importlib.import_module(package_name)

    filter_active = scenarios and scenarios[0] != "all"

    for module_info in pkgutil.iter_modules(pkg.__path__):
        scenario_name = module_info.name
        if filter_active and scenario_name not in scenarios:
            continue

        full_module_name = f"{package_name}.{scenario_name}"

        try:
            module = importlib.import_module(full_module_name)
        except Exception as e:
            print(f"[WARN] Failed to import {full_module_name}: {e}")
            continue

        for attr in vars(module).values():
            
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseTask)
                and attr is not BaseTask
            ):
                task_classes.append(attr)

    return task_classes

TASKS_PACKAGE = "magma_training.scenarios.scenarios"
task_classes = collect_task_classes(TASKS_PACKAGE, ["all"])
for TaskCLS in task_classes:
    try:
        task = TaskCLS()
        task.validate()
    except Exception as e:
        print(f"Failure while loading {TaskCLS.__name__} : {e}")
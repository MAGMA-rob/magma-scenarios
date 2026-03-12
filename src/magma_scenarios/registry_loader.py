import pkgutil
import importlib
from typing import Type

from magma_core.base.tasks import BaseTask, TaskDefinition

from . import envs #For env registry 

TASK_DEFINITION_REGISTRY = {}
TASK_PRESET_REGISTRY = {}

def load_definition(name) -> Type[TaskDefinition]:
    path = TASK_DEFINITION_REGISTRY.get(name,None)
    if path is None:
        raise ImportError(f"Impossible to find {name} in the scenario registry. \
                          Are you sure to use the correct scenario name and task definition name?\n \
                         They should be declared in the __init__.py of the scenario.")
    module_path, class_name = path.split(":")
    return _load_object(module_path, class_name)

def load_preset(name) -> Type[BaseTask]:
    path = TASK_PRESET_REGISTRY.get(name,None)
    if path is None:
        raise ImportError(f"Impossible to find {name} in the scenario registry. \
                          Are you sure to use the correct scenario name and task preset name?\n \
                         They should be declared in the __init__.py of the scenario.")
    module_path, class_name = path.split(":")
    return _load_object(module_path, class_name)


def _load_object(module_path, class_name):

    module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)

    except AttributeError:
        raise ImportError(
            f"Class '{class_name}' not found in module '{module_path}'.\nPlease verify the path registered in the __init__ of the scenario."
        )


def load_scenario_manifests():
    import magma_scenarios.scenarios as scenarios_pkg

    for _, module_name, _ in pkgutil.iter_modules(scenarios_pkg.__path__):

        scenario_module = importlib.import_module(
            f"{scenarios_pkg.__name__}.{module_name}"
        )

        # scenario name (fallback = folder name)
        scenario_name = getattr(
            scenario_module,
            "SCENARIO_NAME",
            module_name
        )

        scenario_base = scenario_module.__name__

        # ----- TASK DEFINITIONS -----
        for local_name, rel_path in getattr(
            scenario_module, "TASK_DEFINITIONS", {}
        ).items():

            module_rel, cls = rel_path.split(":")

            registry_name = f"{scenario_name}.{local_name}"

            TASK_DEFINITION_REGISTRY[registry_name] = (
                f"{scenario_base}.{module_rel}:{cls}"
            )

        # ----- TASK PRESETS -----
        for local_name, rel_path in getattr(
            scenario_module, "TASK_PRESETS", {}
        ).items():

            module_rel, cls = rel_path.split(":")

            registry_name = f"{scenario_name}.{local_name}"

            TASK_PRESET_REGISTRY[registry_name] = (
                f"{scenario_base}.{module_rel}:{cls}"
            )


# Auto load at import
load_scenario_manifests()
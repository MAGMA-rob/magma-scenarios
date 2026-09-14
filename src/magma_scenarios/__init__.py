"""Public loading API for installed MAGMA scenario providers."""

from .manifest import ScenarioManifest
from .registry_loader import (
    get_scenario,
    list_scenarios,
    load_definition,
    load_preset,
    load_skills,
    register_environment,
    __getattr__,
)

__all__ = [
    "ScenarioManifest", "get_scenario", "list_scenarios",
    "load_definition", "load_preset", "load_skills", "register_environment",
    "TASK_DEFINITION_REGISTRY", "TASK_PRESET_REGISTRY", "SKILL_REGISTRY",
]

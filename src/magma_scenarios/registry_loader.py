"""Installed scenario discovery and shared component/environment loading."""

from functools import lru_cache
from importlib import import_module
from importlib.metadata import entry_points
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, TypeVar

from .manifest import ScenarioManifest

if TYPE_CHECKING:
    from magma_core.simulation.skills import BaseSkill
    from magma_core.simulation.tasks import BaseTask, TaskDefinition


Component = TypeVar("Component")
_registration_lock = RLock()
_registered_environments: set[str] = set()


@lru_cache(maxsize=1)
def load_scenario_manifests() -> Mapping[str, ScenarioManifest]:
    """Discover and validate providers once per process, without registering envs."""
    manifests: dict[str, ScenarioManifest] = {}
    providers: dict[str, str] = {}
    environment_providers: dict[str, tuple[str, str]] = {}
    for entry in sorted(entry_points(group="magma.scenarios"), key=lambda item: item.name):
        provider = entry.dist.metadata["Name"] if entry.dist is not None else entry.value
        if entry.name in manifests:
            raise ValueError(
                f"Duplicate scenario {entry.name!r}: {providers[entry.name]!r} and {provider!r}"
            )
        try:
            manifest = entry.load()
        except Exception as error:
            raise ImportError(
                f"Cannot load scenario {entry.name!r} from provider {provider!r}: {entry.value}"
            ) from error
        if not isinstance(manifest, ScenarioManifest):
            raise TypeError(f"Provider {provider!r}: {entry.value!r} is not a ScenarioManifest")
        if manifest.id != entry.name:
            raise ValueError(
                f"Provider {provider!r}: entry point {entry.name!r} differs from manifest {manifest.id!r}"
            )
        for env_id, module in manifest.environments.items():
            previous = environment_providers.get(env_id)
            if previous is not None and previous[0] != module:
                raise ValueError(
                    f"Conflicting environment {env_id!r}: {previous[1]!r} declares "
                    f"{previous[0]!r}, but {provider!r} declares {module!r}"
                )
            environment_providers[env_id] = (module, provider)
        manifests[manifest.id] = manifest
        providers[manifest.id] = provider
    return MappingProxyType(manifests)


def list_scenarios() -> tuple[str, ...]:
    """Return installed scenario IDs in sorted order."""
    return tuple(load_scenario_manifests())


def get_scenario(scenario_id: str) -> ScenarioManifest:
    manifests = load_scenario_manifests()
    try:
        return manifests[scenario_id]
    except KeyError as error:
        raise ValueError(
            f"Unknown scenario {scenario_id!r}. Available: {list(manifests)}. "
            "Install its provider package (pip install -e ./provider for a local checkout)."
        ) from error


def register_environment(environment_id: str) -> None:
    """Register one declared Gym ID without constructing an environment."""
    manifests = load_scenario_manifests()
    owners = [
        manifest for manifest in manifests.values()
        if environment_id in manifest.environments
    ]
    if not owners:
        raise ValueError(f"No installed scenario declares environment {environment_id!r}")
    manifest = owners[0]
    module = manifest.environments[environment_id]
    with _registration_lock:
        if environment_id in _registered_environments:
            return
        try:
            import_module(module)
            from gymnasium.envs.registration import registry

            if environment_id not in registry:
                raise ValueError(f"Module {module!r} did not register Gym ID {environment_id!r}")
        except Exception as error:
            raise ImportError(
                f"Cannot register environment {environment_id!r} for scenario "
                f"{manifest.id!r} using {module!r}"
            ) from error
        _registered_environments.add(environment_id)


def _load_component(
    name: str,
    category: Literal["definitions", "presets", "skills"],
    expected_base: type[Component],
) -> type[Component]:
    if not isinstance(name, str) or name.count('.') != 1:
        raise ValueError(f"Expected 'scenario.Component', got {name!r}")
    scenario_id, local_name = name.split('.', 1)
    manifest = get_scenario(scenario_id)
    entries: Mapping[str, str] = getattr(manifest, category)
    if local_name not in entries:
        raise ValueError(
            f"Unknown {category} component {name!r}. Available: {list(entries)}"
        )
    for environment_id in manifest.environments:
        register_environment(environment_id)
    reference = entries[local_name]
    module_name, class_name = reference.split(':')
    try:
        component: Any = import_module(module_name)
        for attribute in class_name.split('.'):
            component = getattr(component, attribute)
    except Exception as error:
        raise ImportError(f"Cannot load {category} component {name!r} from {reference!r}") from error
    if not isinstance(component, type) or not issubclass(component, expected_base):
        raise TypeError(f"Component {name!r} must be a {expected_base.__name__} subclass")
    return component


def load_definition(name: str) -> "type[TaskDefinition]":
    from magma_core.simulation.tasks import TaskDefinition

    return _load_component(name, "definitions", TaskDefinition)


def load_preset(name: str) -> "type[BaseTask]":
    from magma_core.simulation.tasks import BaseTask

    return _load_component(name, "presets", BaseTask)


def load_skills(target_name: str, skill_names: list[str]) -> "list[type[BaseSkill]]":
    from magma_core.simulation.skills import BaseSkill

    if not isinstance(target_name, str) or target_name.count('.') != 1:
        raise ValueError(f"Expected 'scenario.Component', got {target_name!r}")
    scenario_id = target_name.split('.', 1)[0]
    manifest = get_scenario(scenario_id)
    for environment_id in manifest.environments:
        register_environment(environment_id)
    names = list(manifest.skills) if skill_names == ["all"] else skill_names
    result: list[type[BaseSkill]] = []
    for name in names:
        qualified_name = name if '.' in name else f"{scenario_id}.{name}"
        skill = _load_component(qualified_name, "skills", BaseSkill)
        if skill not in result:
            result.append(skill)
    return result


def __getattr__(name: str) -> Mapping[str, str]:
    categories = {
        "TASK_DEFINITION_REGISTRY": "definitions",
        "TASK_PRESET_REGISTRY": "presets",
        "SKILL_REGISTRY": "skills",
    }
    if name not in categories:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return MappingProxyType({
        f"{manifest.id}.{local_name}": reference
        for manifest in load_scenario_manifests().values()
        for local_name, reference in getattr(manifest, categories[name]).items()
    })

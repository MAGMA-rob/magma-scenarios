def load_definition(name):
    from .registry_loader import load_definition as _load_definition

    return _load_definition(name)


def load_preset(name):
    from .registry_loader import load_preset as _load_preset

    return _load_preset(name)


def __getattr__(name):
    if name in {"TASK_DEFINITION_REGISTRY", "TASK_PRESET_REGISTRY"}:
        from .registry_loader import TASK_DEFINITION_REGISTRY, TASK_PRESET_REGISTRY

        registry = {
            "TASK_DEFINITION_REGISTRY": TASK_DEFINITION_REGISTRY,
            "TASK_PRESET_REGISTRY": TASK_PRESET_REGISTRY,
        }
        return registry[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TASK_DEFINITION_REGISTRY",
    "TASK_PRESET_REGISTRY",
    "load_definition",
    "load_preset",
]

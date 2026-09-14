"""Declarations shared by official and external scenario providers."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ScenarioManifest:
    """Component import references and environment registration modules.

    Component references use ``module:Class``. Environment keys are Gym IDs;
    values are modules that register those IDs when imported.
    """

    id: str
    definitions: Mapping[str, str] = field(default_factory=dict)
    presets: Mapping[str, str] = field(default_factory=dict)
    skills: Mapping[str, str] = field(default_factory=dict)
    environments: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or not self.id
            or '.' in self.id
            or self.id.strip() != self.id
        ):
            raise ValueError("Scenario IDs must be nonempty, trimmed strings without dots")
        for category in ("definitions", "presets", "skills", "environments"):
            entries = getattr(self, category)
            if not isinstance(entries, Mapping):
                raise TypeError(f"Scenario {self.id!r}: {category} must be a mapping")
            for name, reference in entries.items():
                if not isinstance(name, str) or not name or name.strip() != name:
                    raise ValueError(f"Scenario {self.id!r}: invalid {category} name {name!r}")
                if not isinstance(reference, str):
                    raise TypeError(f"Scenario {self.id!r}: {name!r} must reference a module or class")
                parts = reference.split(":")
                if category == "environments":
                    valid = len(parts) == 1
                else:
                    valid = '.' not in name and len(parts) == 2 and all(
                        segment.isidentifier() for segment in parts[-1].split('.')
                    )
                # Importlib supports existing modules such as envs.4_hall_sorting.
                valid = valid and all(
                    segment and segment.replace('_', '').isalnum()
                    for segment in parts[0].split('.')
                )
                if not valid:
                    raise ValueError(
                        f"Scenario {self.id!r}: invalid {category} reference {name!r}: {reference!r}"
                    )
            object.__setattr__(self, category, MappingProxyType(dict(entries)))

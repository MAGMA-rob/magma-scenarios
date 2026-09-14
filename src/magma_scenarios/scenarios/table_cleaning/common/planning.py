from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

from magma_core.domain import Call


def object_type(object_name: str) -> str:
    return object_name.rsplit("_", 1)[0]


def requirements_text(requirements: dict[str, int]) -> str:
    parts = [
        f"{count} {name if count == 1 else name + 's'}"
        for name, count in requirements.items()
    ]
    if not parts:
        raise ValueError("A table requirement cannot be empty.")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"

"""Compatibility declarations derived from the scenario manifest."""

from .manifest import SCENARIO

SCENARIO_NAME = SCENARIO.id
TASK_DEFINITIONS = {
    name: reference.removeprefix(__name__ + ".")
    for name, reference in SCENARIO.definitions.items()
}
TASK_PRESETS = {
    name: reference.removeprefix(__name__ + ".")
    for name, reference in SCENARIO.presets.items()
}
SKILLS = {
    name: reference.removeprefix(__name__ + ".")
    for name, reference in SCENARIO.skills.items()
}

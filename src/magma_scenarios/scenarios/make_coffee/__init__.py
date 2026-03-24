# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

SCENARIO_NAME = "make_coffee"

TASK_DEFINITIONS = {
    "NamedCoffeeDefinition": "coffee_definition:NamedCoffeeDefinition",
}

TASK_PRESETS = {
    "ConstrainedPreset": "coffee_preset:ConstrainedPreset",
    "MultipleUserPreset": "coffee_preset:MultipleUserPreset",
    "TestComposite": "coffee_preset:TestComposite",
}

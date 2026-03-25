# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

SCENARIO_NAME = "press_button"

TASK_DEFINITIONS = {
    "PressButtonDefinition": "button_definition:PressButtonDefinition",
}

TASK_PRESETS = {
    "ButtonPressOrdered": "button_preset:ButtonPressOrdered",
    "ButtonPressNoOrdering": "button_preset:ButtonPressNoOrdering",
    "ButtonPressPreset1": "button_preset:ButtonPressPreset1",
    "ButtonPressPreset2": "button_preset:ButtonPressPreset2",
}

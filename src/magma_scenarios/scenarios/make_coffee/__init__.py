# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

SCENARIO_NAME = "make_coffee"

TASK_DEFINITIONS = {
    "SimpleDefinition": "coffee_definition:SimpleDefinition",
    "TeamDefinition": "coffee_definition:TeamDefinition",
}

TASK_PRESETS = {
    "ConstrainedPreset": "coffee_preset:ConstrainedPreset",
    "TestComposite": "coffee_preset:TestComposite",
    "TeamCoffePreset": "coffee_preset:TeamCoffePreset",

    "SimpBench": "benchmark:CoffeeBenchmark"
}

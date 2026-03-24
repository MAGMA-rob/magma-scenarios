# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

SCENARIO_NAME = "delivery_box"

TASK_DEFINITIONS = {
    "EvolvingRecipeDefinition": "delivery_definition:EvolvingRecipeDefinition",
}

TASK_PRESETS = {
    "BenchDeliveryTask": "simple_preset:BenchDeliveryTask",
    "SimplePreset": "simple_preset:SimplePreset",
}

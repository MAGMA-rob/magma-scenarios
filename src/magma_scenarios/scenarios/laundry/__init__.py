# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

SCENARIO_NAME = "laundry"

TASK_DEFINITIONS = {}

TASK_PRESETS = {
    "LaundryFromDetergentPreset": "laundry_preset:LaundryFromDetergentPreset",
    "LaundryCompatibleClothesPreset": "laundry_preset:LaundryCompatibleClothesPreset",

    #Benshmark
    "SimpBench": "benchmark:LaundryBenchmark"
}

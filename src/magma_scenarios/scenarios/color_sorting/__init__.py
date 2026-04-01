# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

SCENARIO_NAME = "color_sorting"

TASK_DEFINITIONS = {
    "SortingDefinition": "color_sorting_definition:SortingDefinition",
}

TASK_PRESETS = {
    "CleanTablePreset": "simple_preset:CleanTablePreset",

    # Benchmark
    "CSB": "simple_preset:ColorSortingBenchmark"
}

# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

SCENARIO_NAME = "color_sorting"

TASK_DEFINITIONS = {}

TASK_PRESETS = {
    "SortColorWithDetection": "tasks.detection_task:SortColorWithDetection",
    "OrderedSortColorCube": "tasks.main:OrderedSortColorCube",
    "MultipleCubeColorSorting": "tasks.main:MultipleCubeColorSorting",
    "SeqSortColorCube": "tasks.main:SeqSortColorCube",
}

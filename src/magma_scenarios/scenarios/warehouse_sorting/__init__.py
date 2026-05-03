# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

SCENARIO_NAME = "warehouse_sorting"

TASK_DEFINITIONS = {
    "SimpleSortingDefinition": "warehouse_definitions:SimpleSortingDefinition",
    "SortingWithInterdictionsDefinition": "warehouse_definitions:SortingWithInterdictionsDefinition",
    "SortingCategoryDefinition": "warehouse_definitions:SortingCategoryDefinition"
}

TASK_PRESETS = {
    # Preset without manufacturing orders
    "NoManuPreset": "no_manu_preset:NoManuPreset",
    "WarehouseSortingSimpPreset1": "no_manu_preset:WarehouseSortingSimpPreset1",
    "WarehouseSortingSimpAdd": "no_manu_preset:WarehouseSortingSimpAdd",
    "WarehouseSortingSimpAdd2": "no_manu_preset:WarehouseSortingSimpAdd2",
    "WarehouseSortingSimpPreset3": "no_manu_preset:WarehouseSortingSimpPreset3",
    "WarehouseSortingSimpPreset2": "no_manu_preset:WarehouseSortingSimpPreset2",
    "WarehouseSortingSimpInterrupt": "no_manu_preset:WarehouseSortingSimpInterrupt",
    "WarehouseSortingSimpInterdictionPreset1": "no_manu_preset:WarehouseSortingSimpInterdictionPreset1",
    "WarehouseSortingSimpPreset4": "no_manu_preset:WarehouseSortingSimpPreset4",
    "WarehouseSortingSimpInterdictionPreset2": "no_manu_preset:WarehouseSortingSimpInterdictionPreset2",
    "WarehouseSortingSimpPreset5": "no_manu_preset:WarehouseSortingSimpPreset5",
    "WarehouseSortingSimpAdd3": "no_manu_preset:WarehouseSortingSimpAdd3",
    "WarehouseSortingSimpPreset6": "no_manu_preset:WarehouseSortingSimpPreset6",
    "WarehouseSortingSimpPreset7": "no_manu_preset:WarehouseSortingSimpPreset7",

    # Preset with manufacturing orders
    "WarehouseSorting": "preset:WarehouseSorting",
    "WarehouseSortingPreset1": "preset:WarehouseSortingPreset1",


    # Benchmark
    "SimpBench": "benchmark:WS_Simp_Benchmark"
}

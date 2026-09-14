"""Public components provided by this scenario."""

from magma_scenarios.manifest import ScenarioManifest

SCENARIO = ScenarioManifest(
    id='warehouse_sorting',
    definitions={
        "SimpleSortingDefinition": (
            "magma_scenarios.scenarios.warehouse_sorting.warehouse_definitions:SimpleSortingDefinition"
        ),
        "SortingWithInterdictionsDefinition": (
            "magma_scenarios.scenarios.warehouse_sorting.warehouse_definitions:SortingWithInterdictionsDefinition"
        ),
        "SortingCategoryDefinition": (
            "magma_scenarios.scenarios.warehouse_sorting.warehouse_definitions:SortingCategoryDefinition"
        ),
    },
    presets={
        'NoManuPreset': "magma_scenarios.scenarios.warehouse_sorting.no_manu_preset:NoManuPreset",
        "InterruptPreset": (
            "magma_scenarios.scenarios.warehouse_sorting.interrupt_preset:InterruptPreset"
        ),
        "WarehouseSortingSimpPreset1": (
            "magma_scenarios.scenarios.warehouse_sorting.no_manu_preset:WarehouseSortingSimpPreset1"
        ),
        "WarehouseSortingSimpAdd": (
            "magma_scenarios.scenarios.warehouse_sorting.no_manu_preset:WarehouseSortingSimpAdd"
        ),
        "WarehouseSortingSimpAdd2": (
            "magma_scenarios.scenarios.warehouse_sorting.no_manu_preset:WarehouseSortingSimpAdd2"
        ),
        "WarehouseSortingSimpPreset3": (
            "magma_scenarios.scenarios.warehouse_sorting.no_manu_preset:WarehouseSortingSimpPreset3"
        ),
        "WarehouseSortingSimpPreset2": (
            "magma_scenarios.scenarios.warehouse_sorting.no_manu_preset:WarehouseSortingSimpPreset2"
        ),
        "WarehouseSortingSimpInterdictionPreset1": (
            "magma_scenarios.scenarios.warehouse_sorting.no_manu_preset:WarehouseSortingSimpInterdictionPreset1"
        ),
        "WarehouseSortingSimpPreset4": (
            "magma_scenarios.scenarios.warehouse_sorting.no_manu_preset:WarehouseSortingSimpPreset4"
        ),
        "WarehouseSortingSimpInterdictionPreset2": (
            "magma_scenarios.scenarios.warehouse_sorting.no_manu_preset:WarehouseSortingSimpInterdictionPreset2"
        ),
        "WarehouseSortingSimpPreset5": (
            "magma_scenarios.scenarios.warehouse_sorting.no_manu_preset:WarehouseSortingSimpPreset5"
        ),
        "WarehouseSortingSimpAdd3": (
            "magma_scenarios.scenarios.warehouse_sorting.no_manu_preset:WarehouseSortingSimpAdd3"
        ),
        "WarehouseSortingSimpPreset6": (
            "magma_scenarios.scenarios.warehouse_sorting.no_manu_preset:WarehouseSortingSimpPreset6"
        ),
        "WarehouseSortingSimpPreset7": (
            "magma_scenarios.scenarios.warehouse_sorting.no_manu_preset:WarehouseSortingSimpPreset7"
        ),
    },
    skills={
        'cycle': "magma_scenarios.scenarios.warehouse_sorting.warehouse_skill:CycleSkill",
    },
    environments={
        "SortingCubesWarehouse-v1": (
            "magma_scenarios.envs.warehouse_sorting_env.warehouse_sorting_env"
        ),
        "SortingCubesWarehouseTV-v1": (
            "magma_scenarios.envs.warehouse_sorting_env.warehouse_sorting_env"
        ),
    },
)

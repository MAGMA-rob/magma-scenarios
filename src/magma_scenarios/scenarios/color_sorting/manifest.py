"""Public components provided by this scenario."""

from magma_scenarios.manifest import ScenarioManifest

SCENARIO = ScenarioManifest(
    id='color_sorting',
    definitions={
        "SortingDefinition": (
            "magma_scenarios.scenarios.color_sorting.cs_definition:SortingDefinition"
        ),
    },
    presets={
        'CleanTablePreset': "magma_scenarios.scenarios.color_sorting.cs_preset:CleanTablePreset",
        'CrossColorPreset': "magma_scenarios.scenarios.color_sorting.cs_preset:CrossColorPreset",
        'CSB': "magma_scenarios.scenarios.color_sorting.cs_benchmark:ColorSortingBenchmark",
    },
    environments={
        'MixedColorSorting': 'magma_scenarios.envs.six_cubes_two_boxes_on_table.reduced_env',
        "PartialSixCubesTwoBoxesOnTable": (
            "magma_scenarios.envs.six_cubes_two_boxes_on_table.partial_variation"
        ),
        "SixCubesTwoBoxesOnTable": (
            "magma_scenarios.envs.six_cubes_two_boxes_on_table.six_cubes_two_boxes_on_table"
        ),
    },
)

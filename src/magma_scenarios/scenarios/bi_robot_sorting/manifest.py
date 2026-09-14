"""Public components provided by this scenario."""

from magma_scenarios.manifest import ScenarioManifest

SCENARIO = ScenarioManifest(
    id='bi_robot_sorting',
    definitions={
        "BiRobotSortingDefinition": (
            "magma_scenarios.scenarios.bi_robot_sorting.brs_definition:BiRobotSortingDefinition"
        ),
        "SmallMoveSetDefinition": (
            "magma_scenarios.scenarios.bi_robot_sorting.brs_definition:SmallMoveSetDefinition"
        ),
    },
    presets={
        "PrepareRecipe": (
            "magma_scenarios.scenarios.bi_robot_sorting.brs_presets:PrepareRecipePreset"
        ),
        "SortFruitsByType": (
            "magma_scenarios.scenarios.bi_robot_sorting.brs_presets:SortFruitsByTypePreset"
        ),
        "TestSortDamagedObjects": (
            "magma_scenarios.scenarios.bi_robot_sorting.brs_presets:TestSortDamagedObjectsPreset"
        ),
    },
    environments={
        'BiRobotSorting-v1': 'magma_scenarios.envs.bi_robot_sorting.brs_main',
    },
)

"""Public components provided by this scenario."""

from magma_scenarios.manifest import ScenarioManifest

SCENARIO = ScenarioManifest(
    id='hall_sorting',
    definitions={
        "MainDefinition": (
            "magma_scenarios.scenarios.hall_sorting.hs_definition:HallSortingDeliveryDefinition"
        ),
        "SmallHallSortingDefinition": (
            "magma_scenarios.scenarios.hall_sorting.hs_definition:SmallHallSortingDefinition"
        ),
    },
    presets={
        "BalancedTypesByHall": (
            "magma_scenarios.scenarios.hall_sorting.hs_presets:BalancedTypesByHallPreset"
        ),
        "SortTypesByHall": (
            "magma_scenarios.scenarios.hall_sorting.hs_presets:SortTypesByHallPreset"
        ),
    },
    environments={
        '4HallSorting-v1': 'magma_scenarios.envs.4_hall_sorting.main',
    },
)

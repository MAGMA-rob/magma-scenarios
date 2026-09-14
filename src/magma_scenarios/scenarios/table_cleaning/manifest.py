"""Public components provided by this scenario."""

from magma_scenarios.manifest import ScenarioManifest

SCENARIO = ScenarioManifest(
    id='table_cleaning',
    definitions={
        "SimpleDefinition": (
            "magma_scenarios.scenarios.table_cleaning.one_robot.definition:CleanTableDefinition"
        ),
        "AdvancedDefinition": (
            "magma_scenarios.scenarios.table_cleaning.two_robots.definition:AdvancedCleanTableDefinition"
        ),
    },
    presets={
        "SimpleCleanTable": (
            "magma_scenarios.scenarios.table_cleaning.one_robot.preset:TableCleaningPreset"
        ),
        "SimpleSetTable": (
            "magma_scenarios.scenarios.table_cleaning.one_robot.preset:SetTablePreset"
        ),
        "AdvancedCleanTable": (
            "magma_scenarios.scenarios.table_cleaning.two_robots.adct_preset:AdvancedCleaningPreset"
        ),
    },
    skills={
        'deplace': "magma_scenarios.scenarios.table_cleaning.common.clean_table_skill:DeplaceSkill",
    },
    environments={
        'Advanced_Cleaning_Table': 'magma_scenarios.envs.table_cleaning.advanced_env',
        'Cleaning_Table': 'magma_scenarios.envs.table_cleaning.dishwasher_env',
    },
)

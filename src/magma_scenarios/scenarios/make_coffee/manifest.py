"""Public components provided by this scenario."""

from magma_scenarios.manifest import ScenarioManifest

SCENARIO = ScenarioManifest(
    id='make_coffee',
    definitions={
        "SimpleDefinition": (
            "magma_scenarios.scenarios.make_coffee.coffee_definition:SimpleDefinition"
        ),
        'TeamDefinition': "magma_scenarios.scenarios.make_coffee.coffee_definition:TeamDefinition",
    },
    presets={
        "ConstrainedPreset": (
            "magma_scenarios.scenarios.make_coffee.coffee_preset:ConstrainedPreset"
        ),
        'TeamCoffePreset': "magma_scenarios.scenarios.make_coffee.coffee_preset:TeamCoffePreset",
    },
    environments={
        'MakeCoffee-v1': 'magma_scenarios.envs.make_coffee.make_coffee',
    },
)

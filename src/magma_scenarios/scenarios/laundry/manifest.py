"""Public components provided by this scenario."""

from magma_scenarios.manifest import ScenarioManifest

SCENARIO = ScenarioManifest(
    id='laundry',
    definitions={
        'MainDefinition': "magma_scenarios.scenarios.laundry.laundry_def:LaundryDefinition",
    },
    presets={
        "LaundryFromDetergentPreset": (
            "magma_scenarios.scenarios.laundry.laundry_preset:LaundryFromDetergentPreset"
        ),
        "LaundryCompatibleClothesPreset": (
            "magma_scenarios.scenarios.laundry.laundry_preset:LaundryCompatibleClothesPreset"
        ),
        'SimpBench': "magma_scenarios.scenarios.laundry.benchmark:LaundryBenchmark",
    },
    environments={
        'Laundry-v1': 'magma_scenarios.envs.laundry.main',
    },
)

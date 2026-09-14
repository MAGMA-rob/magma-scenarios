"""Public components provided by this scenario."""

from magma_scenarios.manifest import ScenarioManifest

SCENARIO = ScenarioManifest(
    id='packaging',
    definitions={
        "MainDefinition": (
            "magma_scenarios.scenarios.packaging.packaging_definition:PackagingDefinition"
        ),
    },
    presets={
        'DebugPreset': "magma_scenarios.scenarios.packaging.simple_preset:SimplePackagingPreset",
        'SimpBench': "magma_scenarios.scenarios.packaging.benchmark:PackagingBenchmarks",
    },
    environments={
        'Packaging': 'magma_scenarios.envs.packaging_env.main',
    },
)

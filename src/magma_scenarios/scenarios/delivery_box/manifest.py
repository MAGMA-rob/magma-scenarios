"""Public components provided by this scenario."""

from magma_scenarios.manifest import ScenarioManifest

SCENARIO = ScenarioManifest(
    id='delivery_box',
    definitions={
        "MultiOrderDelivery": (
            "magma_scenarios.scenarios.delivery_box.delivery_definition:MultiOrderDeliveryDefinition"
        ),
    },
    presets={
        "BenchDeliveryTask": (
            "magma_scenarios.scenarios.delivery_box.delivery_preset:BenchDeliveryTask"
        ),
        'SimplePreset': "magma_scenarios.scenarios.delivery_box.delivery_preset:SimplePreset",
    },
    skills={
        'cycle': "magma_scenarios.scenarios.delivery_box.delivery_skill:CycleSkill",
    },
    environments={
        'DeliveryBase-v1': 'magma_scenarios.envs.delivery_env.delivery_env',
        'DeliveryBaseTV-v1': 'magma_scenarios.envs.delivery_env.delivery_env',
    },
)

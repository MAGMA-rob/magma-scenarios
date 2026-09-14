"""Public components provided by this scenario."""

from magma_scenarios.manifest import ScenarioManifest

SCENARIO = ScenarioManifest(
    id='advanced_delivery',
    definitions={
        "AdvancedDeliveryDefinition": (
            "magma_scenarios.scenarios.advanced_delivery.ad_definition:AdvancedDeliveryDefinition"
        ),
        "SimpleReceptionDeliveryDefinition": (
            "magma_scenarios.scenarios.advanced_delivery.ad_definition:SimpleReceptionDeliveryDefinition"
        ),
    },
    presets={
        "ParallelReceptionOneDelivery": (
            "magma_scenarios.scenarios.advanced_delivery.ad_presets:ParallelReceptionOneDeliveryPreset"
        ),
    },
    environments={
        'MultiRobotDelivery-v1': 'magma_scenarios.envs.advanced_delivery.main',
    },
)

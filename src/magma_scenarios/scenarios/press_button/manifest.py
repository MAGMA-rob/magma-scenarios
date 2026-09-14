"""Public components provided by this scenario."""

from magma_scenarios.manifest import ScenarioManifest

SCENARIO = ScenarioManifest(
    id='press_button',
    definitions={
        "Definition": (
            "magma_scenarios.scenarios.press_button.button_definition:PressButtonDefinition"
        ),
    },
    presets={
        "ButtonPressOrdered": (
            "magma_scenarios.scenarios.press_button.button_preset:ButtonPressOrdered"
        ),
        "ButtonPressNoOrdering": (
            "magma_scenarios.scenarios.press_button.button_preset:ButtonPressNoOrdering"
        ),
        "ButtonPressPreset1": (
            "magma_scenarios.scenarios.press_button.button_preset:ButtonPressPreset1"
        ),
        "ButtonPressPreset2": (
            "magma_scenarios.scenarios.press_button.button_preset:ButtonPressPreset2"
        ),
    },
    environments={
        'PressButtonBasic-v1': 'magma_scenarios.envs.press_button.press_button',
    },
)

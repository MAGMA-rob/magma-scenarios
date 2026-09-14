from typing import List

from magma_core.simulation.state import RuleRenderer, TaskState

from .button_constraints import (
    BUTTON_RULE_SPECS_KEY,
    PREFIX_RULE_KIND,
    PRECEDENCE_RULE_KIND,
    button_list_to_text,
)


class PressButtonRuleRenderer(RuleRenderer):
    """Project the active button-order rules from the scenario TaskState."""

    def rules(self, state: TaskState) -> List[str]:
        rules: List[str] = []
        for rule_spec in state.properties.get(BUTTON_RULE_SPECS_KEY, []):
            if not isinstance(rule_spec, dict):
                continue
            kind = rule_spec.get("kind")
            if kind == PRECEDENCE_RULE_KIND:
                first_buttons = rule_spec.get("first_buttons", [])
                second_buttons = rule_spec.get("second_buttons", [])
                if first_buttons and second_buttons:
                    rules.append(
                        f"{button_list_to_text(first_buttons)} must be pressed "
                        f"before {button_list_to_text(second_buttons)}."
                    )
            elif kind == PREFIX_RULE_KIND:
                button_name = rule_spec.get("button_name")
                if button_name:
                    rules.append(
                        f"{button_name} must be pressed before any requested "
                        "button sequence."
                    )
        return sorted(rules)

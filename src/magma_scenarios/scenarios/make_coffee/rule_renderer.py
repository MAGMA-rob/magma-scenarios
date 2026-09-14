from collections import defaultdict
from typing import Dict, List

from magma_core.simulation.state import RuleRenderer, TaskState

from .coffee_constraints import (
    COFFEE_PREFERENCE_KEY,
    TEAM_COFFEE_RULES_KEY,
    get_unavailable_coffee_pods,
)
from .coffee_rule_text import (
    coffee_preference_rule,
    team_preference_rule,
    unavailable_coffee_rule,
)


class CoffeeRuleRenderer(RuleRenderer):
    """Project coffee preferences and availability into canonical rules."""

    def rules(self, state: TaskState) -> List[str]:
        rules: List[str] = []
        team_rules = state.relations.get(TEAM_COFFEE_RULES_KEY, {})
        team_covered_preferences = {
            member: rule.get("coffee")
            for rule in team_rules.values()
            for member in rule.get("members", [])
        }

        grouped_preferences: Dict[str, List[str]] = defaultdict(list)
        for person, coffee in state.relations.get(
            COFFEE_PREFERENCE_KEY,
            {},
        ).items():
            if team_covered_preferences.get(person) != coffee:
                grouped_preferences[coffee].append(person)

        for coffee in sorted(grouped_preferences):
            rules.append(coffee_preference_rule(
                sorted(grouped_preferences[coffee]),
                coffee,
            ))
        for team_name, rule in sorted(team_rules.items()):
            rules.append(team_preference_rule(
                team_name,
                rule.get("coffee"),
                default=rule.get("mode") == "default",
            ))
        rules.extend(
            unavailable_coffee_rule(coffee)
            for coffee in sorted(get_unavailable_coffee_pods(state))
        )
        return sorted(rules)


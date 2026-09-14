from collections import defaultdict
from typing import Dict, Iterable, List

from magma_core.simulation.state import RuleRenderer, TaskState


def _join_values(values: Iterable[str]) -> str:
    values = list(values)
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


class WarehouseRuleRenderer(RuleRenderer):
    """Render warehouse sorting state as compact natural-language rules."""

    def rules(self, state: TaskState) -> List[str]:
        rules: List[str] = []
        rules.extend(self._render_object_area_rules(state))
        rules.extend(self._render_object_type_rules(state))
        rules.extend(self._render_type_area_rules(state))
        rules.extend(self._render_forbidden_object_rules(state))
        rules.extend(self._render_forbidden_area_rules(state))
        return sorted(rules)

    def _render_object_area_rules(self, state: TaskState) -> List[str]:
        relation = state.relations.get("object_area", {})
        default_target = state.properties.get(
            "relation_default_targets",
            {},
        ).get("object_area")
        exceptions = self._default_exceptions(
            relation,
            state.attributes.get("objects", []),
            default_target,
        )
        if exceptions is not None:
            if not exceptions:
                return [f"All objects go to {default_target}."]
            rules = [
                self._format_go_to_rule(objects, area)
                for area, objects in exceptions.items()
            ]
            rules.append(f"Remaining objects go to {default_target}.")
            return rules

        grouped = self._group_sources_by_target(relation)
        return [
            self._format_go_to_rule(objects, area)
            for area, objects in grouped.items()
        ]

    def _render_object_type_rules(self, state: TaskState) -> List[str]:
        relation = state.relations.get("object_type", {})
        default_target = state.properties.get(
            "relation_default_targets",
            {},
        ).get("object_type")
        exceptions = self._default_exceptions(
            relation,
            state.attributes.get("objects", []),
            default_target,
        )
        if exceptions is not None:
            if not exceptions:
                return [f"All objects are {default_target}."]
            rules = []
            for object_type, objects in exceptions.items():
                if len(objects) == 1:
                    rules.append(f"{objects[0]} is {object_type}.")
                else:
                    rules.append(
                        f"{_join_values(objects)} are {object_type}."
                    )
            rules.append(f"Remaining objects are {default_target}.")
            return rules

        grouped = self._group_sources_by_target(relation)
        rules = []
        for object_type, objects in grouped.items():
            if len(objects) == 1:
                rules.append(f"{objects[0]} is of type {object_type}.")
            else:
                rules.append(f"{_join_values(objects)} are of type {object_type}.")
        return rules

    def _render_type_area_rules(self, state: TaskState) -> List[str]:
        relation = state.relations.get("type_area", {})
        known_objects = set(state.attributes.get("objects", []))
        known_categories = list(dict.fromkeys(
            object_type
            for object_name, object_type
            in state.relations.get("object_type", {}).items()
            if object_name in known_objects
        ))
        default_target = state.properties.get(
            "relation_default_targets",
            {},
        ).get("type_area")
        exceptions = self._default_exceptions(
            relation,
            known_categories,
            default_target,
        )
        if exceptions is not None:
            if not exceptions:
                rules = [f"All categories go to {default_target}."]
            else:
                rules = []
                for area, categories in exceptions.items():
                    if len(categories) == 1:
                        rules.append(
                            f"Category {categories[0]} goes to {area}."
                        )
                    else:
                        rules.append(
                            f"Categories {_join_values(categories)} go to {area}."
                        )
                rules.append(
                    f"Remaining categories go to {default_target}."
                )
            return rules

        current_relation = {
            category: relation[category]
            for category in known_categories
            if category in relation
        }
        grouped = self._group_sources_by_target(current_relation)
        rules = []
        for area, object_types in grouped.items():
            if len(object_types) == 1:
                rules.append(f"Objects of type {object_types[0]} go to {area}.")
            else:
                rules.append(f"Objects of types {_join_values(object_types)} go to {area}.")
        return rules

    def _render_forbidden_object_rules(self, state: TaskState) -> List[str]:
        forbidden_objects = sorted(
            state.properties.get("forbidden_objects", [])
        )
        if not forbidden_objects:
            return []
        return [f"{_join_values(forbidden_objects)} must not be manipulated."]

    def _render_forbidden_area_rules(self, state: TaskState) -> List[str]:
        forbidden_areas = sorted(
            state.properties.get("forbidden_areas", [])
        )
        if not forbidden_areas:
            return []
        return [f"{_join_values(forbidden_areas)} must not be used."]

    def _group_sources_by_target(self, relation: Dict[str, str]) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = defaultdict(list)
        for source, target in sorted(relation.items()):
            grouped[target].append(source)
        return dict(grouped)

    def _default_exceptions(
        self,
        relation: Dict[str, str],
        universe: Iterable[str],
        default_target: str | None,
    ) -> Dict[str, List[str]] | None:
        sources = sorted(set(universe))
        if (
            default_target is None
            or not sources
        ):
            return None

        return self._group_sources_by_target({
            source: relation.get(source, default_target)
            for source in sources
            if relation.get(source, default_target) != default_target
        })

    def _format_go_to_rule(self, sources: List[str], target: str) -> str:
        if len(sources) == 1:
            return f"{sources[0]} goes to {target}."
        return f"{_join_values(sources)} go to {target}."

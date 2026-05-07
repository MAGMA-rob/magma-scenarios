# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import heapq
from copy import deepcopy
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

from magma_core.base.constraints import BaseConstraint
from magma_core.base.state import TaskState


BUTTON_PRECEDENCE_KEY = "button_precedence"
BUTTON_SEQUENCE_PREFIX_KEY = "button_sequence_prefix"
BUTTON_RULE_SPECS_KEY = "button_rules"
BUTTON_RULE_NEEDS_APPLICATION_KEY = "button_rule_needs_application"
BUTTON_RULE_PENDING_KIND_KEY = "button_rule_pending_kind"
BUTTON_RULE_APPLICATIONS_KEY = "button_rule_applications"

PRECEDENCE_RULE_KIND = "precedence"
PREFIX_RULE_KIND = "prefix"


def available_buttons(state: TaskState) -> List[str]:
    return list(state.attributes.get("objects", []))


def button_list_to_text(buttons: Sequence[str]) -> str:
    if len(buttons) == 1:
        return buttons[0]
    if len(buttons) == 2:
        return f"{buttons[0]} and {buttons[1]}"
    return ", ".join(buttons[:-1]) + f", and {buttons[-1]}"


def _unique(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(values))


def validate_buttons_exist(state: TaskState, buttons: Sequence[str]) -> List[str]:
    known_buttons = available_buttons(state)
    missing = [button for button in buttons if button not in known_buttons]
    if missing:
        raise RuntimeError(f"Unknown buttons {missing} for {known_buttons}")

    unique_buttons = _unique(buttons)
    if not unique_buttons:
        raise RuntimeError("At least one button is required to define an order rule")
    return unique_buttons


def get_rule_specs(state: TaskState) -> List[Dict]:
    rule_specs = state.properties.get(BUTTON_RULE_SPECS_KEY)
    if not isinstance(rule_specs, list):
        rule_specs = []
        state.properties[BUTTON_RULE_SPECS_KEY] = rule_specs
    return rule_specs


def copy_rule_specs(state: TaskState) -> List[Dict]:
    return deepcopy(get_rule_specs(state))


def count_active_rules(state: TaskState) -> int:
    return len(get_rule_specs(state))


def has_any_button_rule(state: TaskState) -> bool:
    return count_active_rules(state) > 0


def can_add_rule(state: TaskState, max_active_rules: int) -> bool:
    return count_active_rules(state) < max_active_rules


def mark_button_rule_pending(state: TaskState, rule_kind: str) -> None:
    state.properties[BUTTON_RULE_NEEDS_APPLICATION_KEY] = True
    state.properties[BUTTON_RULE_PENDING_KIND_KEY] = rule_kind


def button_rule_pending(state: TaskState) -> bool:
    return bool(state.properties.get(BUTTON_RULE_NEEDS_APPLICATION_KEY, False))


def pending_button_rule_kind(state: TaskState) -> Optional[str]:
    if not button_rule_pending(state):
        return None
    return state.properties.get(BUTTON_RULE_PENDING_KIND_KEY)


def clear_button_rule_pending(state: TaskState) -> None:
    if not button_rule_pending(state):
        return
    state.properties[BUTTON_RULE_NEEDS_APPLICATION_KEY] = False
    state.properties[BUTTON_RULE_PENDING_KIND_KEY] = None
    state.properties[BUTTON_RULE_APPLICATIONS_KEY] = (
        state.properties.get(BUTTON_RULE_APPLICATIONS_KEY, 0) + 1
    )


def get_prefix_button(state: TaskState) -> Optional[str]:
    relation = state.relations.get(BUTTON_SEQUENCE_PREFIX_KEY, {})
    if not isinstance(relation, dict):
        return None

    for button_name, is_active in relation.items():
        if is_active:
            return button_name
    return None


def _copy_precedence_graph(state: TaskState) -> Dict[str, List[str]]:
    graph = state.relations.get(BUTTON_PRECEDENCE_KEY, {})
    if not isinstance(graph, dict):
        return {}
    return {
        source: _unique(targets)
        for source, targets in graph.items()
        if isinstance(targets, list)
    }


def _has_path(graph: Dict[str, List[str]], start: str, goal: str) -> bool:
    if start == goal:
        return True

    visited = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current == goal:
            return True
        if current in visited:
            continue
        visited.add(current)
        stack.extend(graph.get(current, []))
    return False


def _would_create_cycle(
    graph: Dict[str, List[str]],
    first_buttons: Sequence[str],
    second_buttons: Sequence[str],
) -> bool:
    return any(
        source == target or _has_path(graph, target, source)
        for source in first_buttons
        for target in second_buttons
    )


def _would_break_prefix(
    graph: Dict[str, List[str]],
    prefix_button: Optional[str],
    second_buttons: Sequence[str],
) -> bool:
    if prefix_button is None:
        return False
    return any(
        target == prefix_button or _has_path(graph, target, prefix_button)
        for target in second_buttons
    )


def _prefix_compatible_with_graph(
    graph: Dict[str, List[str]],
    prefix_button: str,
    buttons: Sequence[str],
) -> bool:
    return all(
        button == prefix_button or not _has_path(graph, button, prefix_button)
        for button in buttons
    )


def _precedence_is_consistent(
    graph: Dict[str, List[str]],
    first_buttons: Sequence[str],
    second_buttons: Sequence[str],
    prefix_button: Optional[str],
) -> bool:
    if not first_buttons or not second_buttons:
        return False
    if set(first_buttons) & set(second_buttons):
        return False
    if _would_create_cycle(graph, first_buttons, second_buttons):
        return False
    return not _would_break_prefix(graph, prefix_button, second_buttons)


def _can_add_precedence(
    state: TaskState,
    graph: Dict[str, List[str]],
    first_buttons: Sequence[str],
    second_buttons: Sequence[str],
) -> bool:
    return _precedence_is_consistent(
        graph,
        first_buttons,
        second_buttons,
        get_prefix_button(state),
    )


def _add_precedence_edges(
    graph: Dict[str, List[str]],
    first_buttons: Sequence[str],
    second_buttons: Sequence[str],
) -> None:
    for source in first_buttons:
        graph.setdefault(source, [])
        for target in second_buttons:
            if target not in graph[source]:
                graph[source].append(target)


def valid_group_candidates(
    state: TaskState,
    max_buttons_per_group: int,
) -> List[Tuple[List[str], List[str]]]:
    buttons = available_buttons(state)
    graph = _copy_precedence_graph(state)
    candidates: List[Tuple[List[str], List[str]]] = []

    for first_size in range(1, min(max_buttons_per_group, len(buttons) - 1) + 1):
        for first_group in combinations(buttons, first_size):
            remaining = [button for button in buttons if button not in first_group]
            for second_size in range(1, min(max_buttons_per_group, len(remaining)) + 1):
                for second_group in combinations(remaining, second_size):
                    first_buttons = list(first_group)
                    second_buttons = list(second_group)
                    already_known = all(
                        target in graph.get(source, [])
                        for source in first_buttons
                        for target in second_buttons
                    )
                    if already_known:
                        continue
                    if _can_add_precedence(state, graph, first_buttons, second_buttons):
                        candidates.append((first_buttons, second_buttons))

    return candidates


def valid_parity_candidates(state: TaskState) -> List[Tuple[List[str], List[str], str]]:
    even_buttons: List[str] = []
    odd_buttons: List[str] = []
    for button in available_buttons(state):
        digits = "".join(char for char in button if char.isdigit())
        if not digits:
            continue
        if int(digits) % 2 == 0:
            even_buttons.append(button)
        else:
            odd_buttons.append(button)

    graph = _copy_precedence_graph(state)
    candidates: List[Tuple[List[str], List[str], str]] = []
    for first_buttons, second_buttons, label in (
        (even_buttons, odd_buttons, "even-first"),
        (odd_buttons, even_buttons, "odd-first"),
    ):
        if not first_buttons or not second_buttons:
            continue
        already_known = all(
            target in graph.get(source, [])
            for source in first_buttons
            for target in second_buttons
        )
        if already_known:
            continue
        if _can_add_precedence(state, graph, first_buttons, second_buttons):
            candidates.append((first_buttons, second_buttons, label))

    return candidates


def available_forget_targets(state: TaskState) -> List[Tuple[str, Optional[str]]]:
    rule_specs = copy_rule_specs(state)
    if not rule_specs:
        return []

    targets: List[Tuple[str, Optional[str]]] = [("all", None)]
    if any(spec.get("kind") == PREFIX_RULE_KIND for spec in rule_specs):
        targets.append(("prefix", None))

    seen_sources = set()
    for spec in rule_specs:
        if spec.get("kind") != PRECEDENCE_RULE_KIND:
            continue
        for button_name in spec.get("first_buttons", []):
            if button_name in seen_sources:
                continue
            seen_sources.add(button_name)
            targets.append(("precedence-source", button_name))

    return targets


def write_rule_specs(state: TaskState, rule_specs: Sequence[Dict]) -> None:
    buttons = available_buttons(state)
    known_buttons = set(buttons)
    graph: Dict[str, List[str]] = {}
    prefix_button: Optional[str] = None
    normalized_specs: List[Dict] = []

    for raw_spec in rule_specs:
        if not isinstance(raw_spec, dict):
            continue

        if raw_spec.get("kind") == PRECEDENCE_RULE_KIND:
            first_buttons = [
                button for button in _unique(raw_spec.get("first_buttons", []))
                if button in known_buttons
            ]
            second_buttons = [
                button for button in _unique(raw_spec.get("second_buttons", []))
                if button in known_buttons and button not in first_buttons
            ]

            if not _precedence_is_consistent(
                graph,
                first_buttons,
                second_buttons,
                prefix_button,
            ):
                continue

            _add_precedence_edges(graph, first_buttons, second_buttons)
            normalized_specs.append({
                "kind": PRECEDENCE_RULE_KIND,
                "rule_name": raw_spec.get("rule_name", "group-order"),
                "first_buttons": first_buttons,
                "second_buttons": second_buttons,
            })
            continue

        if raw_spec.get("kind") == PREFIX_RULE_KIND:
            button_name = raw_spec.get("button_name")
            if button_name not in known_buttons:
                continue
            if not _prefix_compatible_with_graph(graph, button_name, buttons):
                continue

            prefix_button = button_name
            normalized_specs = [
                spec for spec in normalized_specs
                if spec.get("kind") != PREFIX_RULE_KIND
            ]
            normalized_specs.append({
                "kind": PREFIX_RULE_KIND,
                "button_name": button_name,
            })

    state.properties[BUTTON_RULE_SPECS_KEY] = normalized_specs
    state.relations[BUTTON_PRECEDENCE_KEY] = graph
    state.relations[BUTTON_SEQUENCE_PREFIX_KEY] = (
        {prefix_button: True}
        if prefix_button is not None
        else {}
    )


def apply_prefix_rule(state: TaskState, ordered_buttons: Sequence[str]) -> List[str]:
    prefix_button = get_prefix_button(state)
    buttons = _unique(ordered_buttons)
    if prefix_button is None:
        return buttons

    buttons = [button for button in buttons if button != prefix_button]
    return [prefix_button] + buttons


def resolve_prefix_only_order(
    state: TaskState,
    requested_buttons: Sequence[str],
) -> List[str]:
    validate_buttons_exist(state, requested_buttons)
    return apply_prefix_rule(state, requested_buttons)


def resolve_order_with_rules(
    state: TaskState,
    requested_buttons: Sequence[str],
) -> List[str]:
    requested_buttons = validate_buttons_exist(state, requested_buttons)
    selected_buttons = set(requested_buttons)
    graph = _copy_precedence_graph(state)

    adjacency = {button: [] for button in requested_buttons}
    indegree = {button: 0 for button in requested_buttons}
    input_order = {button: index for index, button in enumerate(requested_buttons)}

    for source in requested_buttons:
        for target in graph.get(source, []):
            if target not in selected_buttons:
                continue
            adjacency[source].append(target)
            indegree[target] += 1

    ready: List[Tuple[int, str]] = []
    for button in requested_buttons:
        if indegree[button] == 0:
            heapq.heappush(ready, (input_order[button], button))

    ordered_buttons = []
    while ready:
        _, button = heapq.heappop(ready)
        ordered_buttons.append(button)

        for target in adjacency[button]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, (input_order[target], target))

    if len(ordered_buttons) != len(requested_buttons):
        raise RuntimeError("Failed to resolve a consistent button order from the active rules")

    return apply_prefix_rule(state, ordered_buttons)


def button_rule_application_seeds(
    state: TaskState,
    rule_kind: Optional[str] = None,
) -> List[List[str]]:
    buttons = available_buttons(state)
    known_buttons = set(buttons)
    seeds: List[List[str]] = []

    for spec in copy_rule_specs(state):
        kind = spec.get("kind")
        if rule_kind is not None and kind != rule_kind:
            continue

        if kind == PREFIX_RULE_KIND:
            prefix_button = get_prefix_button(state)
            if prefix_button is None:
                continue
            seeds.extend([[button] for button in buttons if button != prefix_button])

        if kind == PRECEDENCE_RULE_KIND:
            first_buttons = [
                button for button in spec.get("first_buttons", [])
                if button in known_buttons
            ]
            second_buttons = [
                button for button in spec.get("second_buttons", [])
                if button in known_buttons and button not in first_buttons
            ]
            seeds.extend(
                [first_button, second_button]
                for first_button in first_buttons
                for second_button in second_buttons
            )

    return seeds


def request_uses_button_rule(state: TaskState, requested_buttons: Sequence[str]) -> bool:
    requested_buttons = validate_buttons_exist(state, requested_buttons)
    selected_buttons = set(requested_buttons)

    prefix_button = get_prefix_button(state)
    if prefix_button is not None and any(button != prefix_button for button in requested_buttons):
        return True

    graph = _copy_precedence_graph(state)
    return any(
        source in selected_buttons and target in selected_buttons
        for source, targets in graph.items()
        for target in targets
    )


def request_uses_pending_button_rule(state: TaskState, requested_buttons: Sequence[str]) -> bool:
    pending_kind = pending_button_rule_kind(state)
    if pending_kind is None:
        return request_uses_button_rule(state, requested_buttons)

    selected_buttons = set(validate_buttons_exist(state, requested_buttons))
    if pending_kind == PREFIX_RULE_KIND:
        prefix_button = get_prefix_button(state)
        return prefix_button is not None and any(
            button != prefix_button
            for button in selected_buttons
        )

    if pending_kind == PRECEDENCE_RULE_KIND:
        graph = _copy_precedence_graph(state)
        return any(
            source in selected_buttons and target in selected_buttons
            for source, targets in graph.items()
            for target in targets
        )

    return request_uses_button_rule(state, requested_buttons)


class ButtonPrecedenceConstraint(BaseConstraint):
    """Store a permanent order rule between two button groups."""

    def __init__(
        self,
        first_buttons: Sequence[str],
        second_buttons: Sequence[str],
        rule_name: str = "group-order",
    ) -> None:
        super().__init__()
        self.first_buttons = _unique(first_buttons)
        self.second_buttons = _unique(second_buttons)
        self.rule_name = rule_name

        if not self.first_buttons or not self.second_buttons:
            raise ValueError("A precedence rule needs two non-empty button groups")

    def apply(self, state: TaskState):
        super().apply(state)
        first_buttons = validate_buttons_exist(state, self.first_buttons)
        second_buttons = validate_buttons_exist(state, self.second_buttons)
        graph = _copy_precedence_graph(state)

        if not _can_add_precedence(state, graph, first_buttons, second_buttons):
            raise RuntimeError(
                f"Applying {self.__class__.__name__} would create an invalid button order"
            )

        rule_specs = copy_rule_specs(state)
        rule_specs.append({
            "kind": PRECEDENCE_RULE_KIND,
            "rule_name": self.rule_name,
            "first_buttons": first_buttons,
            "second_buttons": second_buttons,
        })
        write_rule_specs(state, rule_specs)

    def outdated(self, state: TaskState) -> bool:
        known_buttons = set(available_buttons(state))
        return any(
            button not in known_buttons
            for button in self.first_buttons + self.second_buttons
        )


class ButtonSequencePrefixConstraint(BaseConstraint):
    """Store the button that must be pressed before requested sequences."""

    def __init__(self, button_name: str) -> None:
        super().__init__()
        self.button_name = button_name

    def apply(self, state: TaskState):
        super().apply(state)
        validate_buttons_exist(state, [self.button_name])

        graph = _copy_precedence_graph(state)
        if not _prefix_compatible_with_graph(graph, self.button_name, available_buttons(state)):
            raise RuntimeError(
                f"Applying {self.__class__.__name__} would contradict existing button rules"
            )

        rule_specs = [
            spec for spec in copy_rule_specs(state)
            if spec.get("kind") != PREFIX_RULE_KIND
        ]
        rule_specs.append({
            "kind": PREFIX_RULE_KIND,
            "button_name": self.button_name,
        })
        write_rule_specs(state, rule_specs)

    def outdated(self, state: TaskState) -> bool:
        return self.button_name not in available_buttons(state)


class ForgetButtonRulesConstraint(BaseConstraint):
    """Clear all button rules or a targeted subset of them."""

    def __init__(self, mode: str = "all", button_name: Optional[str] = None) -> None:
        super().__init__()
        self.mode = mode
        self.button_name = button_name

    def apply(self, state: TaskState):
        super().apply(state)
        rule_specs = copy_rule_specs(state)

        if self.mode == "all":
            write_rule_specs(state, [])
            state.properties[BUTTON_RULE_NEEDS_APPLICATION_KEY] = False
            state.properties[BUTTON_RULE_PENDING_KIND_KEY] = None
            return

        if self.mode == "prefix":
            write_rule_specs(
                state,
                [spec for spec in rule_specs if spec.get("kind") != PREFIX_RULE_KIND],
            )
            state.properties[BUTTON_RULE_NEEDS_APPLICATION_KEY] = False
            state.properties[BUTTON_RULE_PENDING_KIND_KEY] = None
            return

        if self.mode == "precedence-source":
            if self.button_name is None:
                raise RuntimeError("button_name must be defined for precedence-source forgetting")

            updated_specs = []
            for spec in rule_specs:
                if spec.get("kind") != PRECEDENCE_RULE_KIND:
                    updated_specs.append(spec)
                    continue

                first_buttons = [
                    button for button in spec.get("first_buttons", [])
                    if button != self.button_name
                ]
                if len(first_buttons) == len(spec.get("first_buttons", [])):
                    updated_specs.append(spec)
                    continue
                if not first_buttons:
                    continue

                spec_copy = deepcopy(spec)
                spec_copy["first_buttons"] = first_buttons
                updated_specs.append(spec_copy)

            write_rule_specs(state, updated_specs)
            state.properties[BUTTON_RULE_NEEDS_APPLICATION_KEY] = False
            state.properties[BUTTON_RULE_PENDING_KIND_KEY] = None
            return

        raise RuntimeError(f"Unknown forgetting mode: {self.mode}")

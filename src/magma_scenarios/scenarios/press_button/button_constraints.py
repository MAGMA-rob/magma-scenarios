# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import heapq
from copy import deepcopy
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from magma_core.base.constraints import BaseConstraint
from magma_core.base.state import TaskState


BUTTON_PRECEDENCE_KEY = "button_precedence"
BUTTON_SEQUENCE_PREFIX_KEY = "button_sequence_prefix"
BUTTON_RULE_SPECS_KEY = "button_rules"


def available_buttons(state: TaskState) -> List[str]:
    """Read the current button list from the task state."""
    return list(state.attributes.get("objects", []))


def validate_buttons_exist(state: TaskState, buttons: Sequence[str]) -> List[str]:
    """Validate button names and return a deduplicated list in input order."""
    known_buttons = available_buttons(state)
    missing = [button for button in buttons if button not in known_buttons]
    if missing:
        raise RuntimeError(
            f"Unknown buttons {missing} for {state.attributes.get('objects', [])}"
        )

    unique_buttons = list(dict.fromkeys(buttons))
    if not unique_buttons:
        raise RuntimeError("At least one button is required to define an order constraint")
    return unique_buttons


def button_list_to_text(buttons: Sequence[str]) -> str:
    """Format button names for natural-language instructions."""
    if len(buttons) == 1:
        return buttons[0]
    if len(buttons) == 2:
        return f"{buttons[0]} and {buttons[1]}"
    return ", ".join(buttons[:-1]) + f", and {buttons[-1]}"


def get_rule_specs(state: TaskState) -> List[Dict]:
    """Return the active conceptual rules, creating the list on first use."""
    rule_specs = state.properties.get(BUTTON_RULE_SPECS_KEY)
    if not isinstance(rule_specs, list):
        rule_specs = []
        state.properties[BUTTON_RULE_SPECS_KEY] = rule_specs
    return rule_specs


def copy_rule_specs(state: TaskState) -> List[Dict]:
    """Return a deep copy of the active rule specs."""
    return deepcopy(get_rule_specs(state))


def count_active_rules(state: TaskState) -> int:
    """Number of conceptual button rules currently active."""
    return len(get_rule_specs(state))


def can_add_rule(state: TaskState, max_active_rules: int) -> bool:
    """Return True if a new permanent rule can still be sampled."""
    return count_active_rules(state) < max_active_rules


def get_precedence_graph(state: TaskState) -> Dict[str, List[str]]:
    """Return the stored precedence graph, creating it on first use."""
    graph = state.relations.get(BUTTON_PRECEDENCE_KEY)
    if not isinstance(graph, dict):
        graph = {}
        state.relations[BUTTON_PRECEDENCE_KEY] = graph
    return graph


def copy_precedence_graph(state: TaskState) -> Dict[str, List[str]]:
    """Return a sanitized copy used to validate candidates before mutation."""
    graph = state.relations.get(BUTTON_PRECEDENCE_KEY, {})
    return {
        source: list(dict.fromkeys(targets))
        for source, targets in graph.items()
        if isinstance(targets, list)
    }


def get_prefix_button(state: TaskState) -> Optional[str]:
    """Return the active prefix button if one exists."""
    relation = state.relations.get(BUTTON_SEQUENCE_PREFIX_KEY, {})
    if not isinstance(relation, dict):
        return None

    for button_name, is_active in relation.items():
        if is_active:
            return button_name
    return None


def has_path(graph: Dict[str, List[str]], start: str, goal: str) -> bool:
    """Depth-first search: do existing rules already imply start -> goal?"""
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


def would_create_cycle(
    graph: Dict[str, List[str]],
    first_buttons: Sequence[str],
    second_buttons: Sequence[str],
) -> bool:
    """Return True if adding first_buttons -> second_buttons would contradict the graph."""
    for source in first_buttons:
        for target in second_buttons:
            if source == target or has_path(graph, target, source):
                return True
    return False


def would_break_prefix_rule(
    state: TaskState,
    graph: Dict[str, List[str]],
    first_buttons: Sequence[str],
    second_buttons: Sequence[str],
) -> bool:
    """Return True if a new precedence rule would force another button before the prefix."""
    prefix_button = get_prefix_button(state)
    if prefix_button is None:
        return False

    for target in second_buttons:
        if target == prefix_button or has_path(graph, target, prefix_button):
            return True
    return False


def all_edges_present(
    graph: Dict[str, List[str]],
    first_buttons: Sequence[str],
    second_buttons: Sequence[str],
) -> bool:
    """Return True when a candidate rule is already fully encoded in state."""
    return all(
        target in graph.get(source, [])
        for source in first_buttons
        for target in second_buttons
    )


def iter_group_candidates(
    buttons: Sequence[str],
    max_buttons_per_group: int,
) -> Iterable[Tuple[List[str], List[str]]]:
    """Enumerate all disjoint button-group pairs that could define an order rule."""
    nb_buttons = len(buttons)
    if nb_buttons < 2:
        return []

    max_first_group = min(max_buttons_per_group, nb_buttons - 1)
    for first_group_size in range(1, max_first_group + 1):
        for first_group in combinations(buttons, first_group_size):
            remaining = [button for button in buttons if button not in first_group]
            max_second_group = min(max_buttons_per_group, len(remaining))
            for second_group_size in range(1, max_second_group + 1):
                for second_group in combinations(remaining, second_group_size):
                    yield list(first_group), list(second_group)


def valid_group_candidates(
    state: TaskState,
    max_buttons_per_group: int,
) -> List[Tuple[List[str], List[str]]]:
    """Keep only group rules that are new, acyclic, and prefix-compatible."""
    buttons = available_buttons(state)
    graph = copy_precedence_graph(state)
    candidates = []

    for first_group, second_group in iter_group_candidates(buttons, max_buttons_per_group):
        if all_edges_present(graph, first_group, second_group):
            continue
        if would_create_cycle(graph, first_group, second_group):
            continue
        if would_break_prefix_rule(state, graph, first_group, second_group):
            continue
        candidates.append((first_group, second_group))

    return candidates


def button_number(button_name: str) -> Optional[int]:
    """Extract the numeric suffix of names like sw0, sw1, ..."""
    digits = "".join(char for char in button_name if char.isdigit())
    if not digits:
        return None
    return int(digits)


def split_even_odd_buttons(buttons: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Partition known buttons into even and odd groups using their numeric suffix."""
    even_buttons = []
    odd_buttons = []

    for button in buttons:
        cur_number = button_number(button)
        if cur_number is None:
            continue
        if cur_number % 2 == 0:
            even_buttons.append(button)
        else:
            odd_buttons.append(button)

    return even_buttons, odd_buttons


def valid_parity_candidates(
    state: TaskState,
) -> List[Tuple[List[str], List[str], str]]:
    """Build valid even-first / odd-first candidates from the current state."""
    even_buttons, odd_buttons = split_even_odd_buttons(available_buttons(state))
    graph = copy_precedence_graph(state)
    candidates = []

    for first_group, second_group, label in (
        (even_buttons, odd_buttons, "even-first"),
        (odd_buttons, even_buttons, "odd-first"),
    ):
        if not first_group or not second_group:
            continue
        if all_edges_present(graph, first_group, second_group):
            continue
        if would_create_cycle(graph, first_group, second_group):
            continue
        if would_break_prefix_rule(state, graph, first_group, second_group):
            continue
        candidates.append((first_group, second_group, label))

    return candidates


def has_any_button_rule(state: TaskState) -> bool:
    """Used by the forget request to know if there is something to clear."""
    return count_active_rules(state) > 0


def available_forget_targets(state: TaskState) -> List[Tuple[str, Optional[str]]]:
    """Return all valid forget actions from the current rule state."""
    rule_specs = copy_rule_specs(state)
    if not rule_specs:
        return []

    targets: List[Tuple[str, Optional[str]]] = [("all", None)]

    if any(spec.get("kind") == "prefix" for spec in rule_specs):
        targets.append(("prefix", None))

    seen_sources = set()
    for spec in rule_specs:
        if spec.get("kind") != "precedence":
            continue
        for button_name in spec.get("first_buttons", []):
            if button_name not in seen_sources:
                seen_sources.add(button_name)
                targets.append(("precedence-source", button_name))

    return targets


def count_forgettable_targets(state: TaskState) -> int:
    """Convenience helper used by request weighting."""
    return len(available_forget_targets(state))


def _prefix_compatible_with_graph(graph: Dict[str, List[str]], prefix_button: str, buttons: Sequence[str]) -> bool:
    """A prefix button must not be constrained to come after another button."""
    return all(
        source_button == prefix_button or not has_path(graph, source_button, prefix_button)
        for source_button in buttons
    )


def _normalized_precedence_spec(
    state: TaskState,
    first_buttons: Sequence[str],
    second_buttons: Sequence[str],
    rule_name: str,
) -> Dict:
    return {
        "kind": "precedence",
        "rule_name": rule_name,
        "first_buttons": validate_buttons_exist(state, first_buttons),
        "second_buttons": validate_buttons_exist(state, second_buttons),
    }


def _normalized_prefix_spec(state: TaskState, button_name: str) -> Dict:
    validate_buttons_exist(state, [button_name])
    return {
        "kind": "prefix",
        "button_name": button_name,
    }


def write_rule_specs(state: TaskState, rule_specs: Sequence[Dict]) -> None:
    """Rebuild the derived button relations from a conceptual rule list."""
    known_buttons = available_buttons(state)
    known_button_set = set(known_buttons)

    new_rule_specs: List[Dict] = []
    new_graph: Dict[str, List[str]] = {}
    prefix_button: Optional[str] = None

    for raw_spec in rule_specs:
        if not isinstance(raw_spec, dict):
            continue

        kind = raw_spec.get("kind")
        if kind == "precedence":
            first_buttons = [
                button
                for button in dict.fromkeys(raw_spec.get("first_buttons", []))
                if button in known_button_set
            ]
            second_buttons = [
                button
                for button in dict.fromkeys(raw_spec.get("second_buttons", []))
                if button in known_button_set and button not in first_buttons
            ]

            if not first_buttons or not second_buttons:
                continue
            if would_create_cycle(new_graph, first_buttons, second_buttons):
                continue
            if prefix_button is not None and any(
                target == prefix_button or has_path(new_graph, target, prefix_button)
                for target in second_buttons
            ):
                continue

            for source in first_buttons:
                new_graph.setdefault(source, [])
                for target in second_buttons:
                    if target not in new_graph[source]:
                        new_graph[source].append(target)

            new_rule_specs.append({
                "kind": "precedence",
                "rule_name": raw_spec.get("rule_name", "group-order"),
                "first_buttons": first_buttons,
                "second_buttons": second_buttons,
            })

        elif kind == "prefix":
            button_name = raw_spec.get("button_name")
            if button_name not in known_button_set:
                continue
            if not _prefix_compatible_with_graph(new_graph, button_name, known_buttons):
                continue

            prefix_button = button_name
            new_rule_specs = [
                spec for spec in new_rule_specs
                if spec.get("kind") != "prefix"
            ]
            new_rule_specs.append({
                "kind": "prefix",
                "button_name": button_name,
            })

    state.properties[BUTTON_RULE_SPECS_KEY] = new_rule_specs
    state.relations[BUTTON_PRECEDENCE_KEY] = new_graph
    state.relations[BUTTON_SEQUENCE_PREFIX_KEY] = (
        {prefix_button: True}
        if prefix_button is not None
        else {}
    )


def apply_prefix_rule(state: TaskState, ordered_buttons: Sequence[str]) -> List[str]:
    """Place the prefix button first, without duplicating it if it is already present."""
    prefix_button = get_prefix_button(state)
    normalized_buttons = list(dict.fromkeys(ordered_buttons))
    if prefix_button is None:
        return normalized_buttons

    normalized_buttons = [button for button in normalized_buttons if button != prefix_button]
    return [prefix_button] + normalized_buttons


def resolve_prefix_only_order(
    state: TaskState,
    requested_buttons: Sequence[str],
) -> List[str]:
    """Exact-order requests only honor the prefix rule, not implicit precedence rules."""
    validate_buttons_exist(state, requested_buttons)
    return apply_prefix_rule(state, requested_buttons)


def resolve_order_with_rules(
    state: TaskState,
    requested_buttons: Sequence[str],
) -> List[str]:
    """Resolve the effective order using precedence rules, then apply the prefix rule."""
    requested_buttons = validate_buttons_exist(state, requested_buttons)
    selected_set = set(requested_buttons)
    graph = copy_precedence_graph(state)

    adjacency = {button: [] for button in requested_buttons}
    indegree = {button: 0 for button in requested_buttons}
    order_index = {button: index for index, button in enumerate(requested_buttons)}

    for source in requested_buttons:
        for target in graph.get(source, []):
            if target not in selected_set:
                continue
            adjacency[source].append(target)
            indegree[target] += 1

    available_heap: List[Tuple[int, str]] = []
    for button in requested_buttons:
        if indegree[button] == 0:
            heapq.heappush(available_heap, (order_index[button], button))

    ordered_buttons = []
    while available_heap:
        _, button = heapq.heappop(available_heap)
        ordered_buttons.append(button)

        for target in adjacency[button]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(available_heap, (order_index[target], target))

    if len(ordered_buttons) != len(requested_buttons):
        raise RuntimeError("Failed to resolve a consistent button order from the active rules")

    return apply_prefix_rule(state, ordered_buttons)


class ButtonPrecedenceConstraint(BaseConstraint):
    """Store precedence edges that later requests can turn into an exact order."""

    def __init__(
        self,
        first_buttons: Sequence[str],
        second_buttons: Sequence[str],
        rule_name: str = "group-order",
    ) -> None:
        super().__init__()
        self.first_buttons = list(dict.fromkeys(first_buttons))
        self.second_buttons = list(dict.fromkeys(second_buttons))
        self.rule_name = rule_name

        if not self.first_buttons or not self.second_buttons:
            raise ValueError("A precedence constraint needs two non-empty button groups")

    def apply(self, state: TaskState):
        super().apply(state)

        first_buttons = validate_buttons_exist(state, self.first_buttons)
        second_buttons = validate_buttons_exist(state, self.second_buttons)

        if set(first_buttons) & set(second_buttons):
            raise RuntimeError("The same button can not appear on both sides of an order rule")

        graph = copy_precedence_graph(state)
        if would_create_cycle(graph, first_buttons, second_buttons):
            raise RuntimeError(
                f"Applying {self.__class__.__name__} would create a contradictory button order"
            )
        if would_break_prefix_rule(state, graph, first_buttons, second_buttons):
            raise RuntimeError(
                f"Applying {self.__class__.__name__} would contradict the active prefix rule"
            )

        rule_specs = copy_rule_specs(state)
        rule_specs.append(
            _normalized_precedence_spec(state, first_buttons, second_buttons, self.rule_name)
        )
        write_rule_specs(state, rule_specs)

    def outdated(self, state: TaskState) -> bool:
        available = set(available_buttons(state))
        return any(button not in available for button in self.first_buttons + self.second_buttons)


class ButtonSequencePrefixConstraint(BaseConstraint):
    """Define one button that must be pressed before the requested sequence."""

    def __init__(self, button_name: str) -> None:
        super().__init__()
        self.button_name = button_name

    def apply(self, state: TaskState):
        super().apply(state)
        validate_buttons_exist(state, [self.button_name])

        graph = copy_precedence_graph(state)
        if not _prefix_compatible_with_graph(graph, self.button_name, available_buttons(state)):
            raise RuntimeError(
                f"Applying {self.__class__.__name__} would contradict existing precedence rules"
            )

        rule_specs = [
            spec for spec in copy_rule_specs(state)
            if spec.get("kind") != "prefix"
        ]
        rule_specs.append(_normalized_prefix_spec(state, self.button_name))
        write_rule_specs(state, rule_specs)

    def outdated(self, state: TaskState) -> bool:
        return self.button_name not in available_buttons(state)


class ForgetButtonRulesConstraint(BaseConstraint):
    """Clear all rules or a targeted subset of them."""

    def __init__(self, mode: str = "all", button_name: Optional[str] = None) -> None:
        super().__init__()
        self.mode = mode
        self.button_name = button_name

    def apply(self, state: TaskState):
        super().apply(state)

        rule_specs = copy_rule_specs(state)

        if self.mode == "all":
            write_rule_specs(state, [])
            return

        if self.mode == "prefix":
            write_rule_specs(
                state,
                [spec for spec in rule_specs if spec.get("kind") != "prefix"],
            )
            return

        if self.mode == "precedence-source":
            if self.button_name is None:
                raise RuntimeError("button_name must be defined for precedence-source forgetting")

            updated_specs = []
            for spec in rule_specs:
                if spec.get("kind") != "precedence":
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
            return

        raise RuntimeError(f"Unknown forgetting mode: {self.mode}")

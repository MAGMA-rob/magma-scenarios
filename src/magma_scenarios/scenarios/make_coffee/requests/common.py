import random
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

from magma_core.utils.text_utils import join_with_and, is_or_are


CoffeeOrderItem = Tuple[str, Optional[str]]


def unavailable_coffee_message(blocked_pods: Sequence[str]) -> str:
    pods = list(blocked_pods)
    return f"{join_with_and(pods)} coffee {is_or_are(pods)} unavailable."


def unavailable_coffee_verification(blocked_pods: Sequence[str]) -> str:
    return f"The model must refuse because {unavailable_coffee_message(blocked_pods)[:-1]}"


def missing_preference_verification(missing_names: Sequence[str]) -> str:
    names = list(missing_names)
    joined_names = join_with_and(names)
    if len(names) == 1:
        return f"The model must inform that {joined_names} does not have any coffee preference OR ask for their coffee preference."
    return f"The model must inform that {joined_names} do not have any coffee preference OR ask for their coffee preferences."


def preference_resolution(missing_assignment: Dict[str, str]) -> str:
    parts = [f"{name} wants a {pod} coffee" for name, pod in missing_assignment.items()]
    return join_with_and(parts) + "." if parts else ""


def ordered_coffee_instruction(names: Sequence[str]) -> str:
    normalized_names = list(names)
    if len(normalized_names) == 1:
        return random.choice((f"Can you serve a coffee for {normalized_names[0]}?", f"Please make the coffee for {normalized_names[0]}."))
    parts = [f"first the coffee for {normalized_names[0]}"]
    parts.extend(f"then the one for {name}" for name in normalized_names[1:])
    return f"Please make {join_with_and(parts)}."


def unavailable_people_message(blocked_names: Sequence[str], capsule_by_name: Dict[str, str]) -> str:
    names_by_pod = defaultdict(list)
    for name in blocked_names:
        names_by_pod[capsule_by_name[name]].append(name)
    parts = [f"{join_with_and(names)} cannot be served because {pod} coffee is unavailable" for pod, names in names_by_pod.items()]
    return join_with_and(parts) + "."


def unavailable_people_verification(blocked_names: Sequence[str], capsule_by_name: Dict[str, str]) -> str:
    return "The model must refuse and inform that " + unavailable_people_message(blocked_names, capsule_by_name)


def cancel_coffee_instruction(blocked_names: Sequence[str], served_names: Sequence[str]) -> str:
    return f"Do not make the coffee for {join_with_and(list(blocked_names))}. Please prepare only the coffee for {join_with_and(list(served_names))}."


def substitute_coffee_instruction(
    blocked_names: Sequence[str],
    replacement_pod: str,
    served_names: Sequence[str],
    *,
    ordered: bool = True,
) -> str:
    continuation = (
        ordered_coffee_instruction(served_names)
        if ordered
        else f"Please make coffee for {join_with_and(list(served_names))}."
    )
    return f"Use {replacement_pod} coffee instead for {join_with_and(list(blocked_names))}. {continuation}"

from typing import Sequence

from magma_core.utils.text_utils import join_with_and


def coffee_preference_rule(people: Sequence[str], coffee: str) -> str:
    normalized_people = list(people)
    verb = "likes" if len(normalized_people) == 1 else "like"
    return f"{join_with_and(normalized_people)} {verb} {coffee} coffee."


def team_preference_rule(
    team_name: str,
    coffee: str,
    *,
    default: bool,
) -> str:
    suffix = " by default" if default else ""
    return f"Team {team_name} prefers {coffee} coffee{suffix}."


def unavailable_coffee_rule(coffee: str) -> str:
    return f"{coffee} coffee is unavailable."

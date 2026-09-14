from typing import Iterable, List


from ...common.attributes import dishware, food
from ...common.planning import object_type


def available_requirement_slots(active_objects: Iterable[str]) -> List[str]:
    eligible = set(food + dishware)
    return [
        object_type(object_name)
        for object_name in active_objects
        if object_name in eligible
    ]

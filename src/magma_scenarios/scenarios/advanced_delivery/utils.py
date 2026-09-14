from typing import Dict, Iterable, List, Optional

from .attributes import PRODUCT_TYPES


def get_pose(entry):
    if isinstance(entry, dict):
        return entry["pose"]

    return entry


def product_type_from_name(object_name: str) -> Optional[str]:
    product_type, separator, instance_index = (object_name.rpartition("_"))

    if (
        not separator
        or product_type not in PRODUCT_TYPES
        or not instance_index.isdigit()
    ):
        return None

    return product_type


def group_product_objects(extra: Dict, excluded_objects: Iterable[str] = ()) -> Dict[str, List[str]]:
    excluded = set(excluded_objects)

    objects_by_type = {
        product_type: []
        for product_type in PRODUCT_TYPES
    }

    for object_name, entry in extra.items():
        if object_name in excluded:
            continue

        if not isinstance(entry, dict) or "pose" not in entry:
            continue

        product_type = product_type_from_name(object_name)

        if product_type is None:
            continue

        objects_by_type[product_type].append(object_name)

    return objects_by_type
import random
from collections import Counter

PACKAGE_NAMES = ["package_1","package_2","package_3","package_4"]

att = {"product_type":["coca","icetea","brets","donut"],
       "slot": ["products_slot", "packages_slot"],
       "package": PACKAGE_NAMES }

PRODUCT_TYPES = ["coca", "icetea", "brets", "donut"]


def _sample_table_objects() -> list[str]:
    available_types = [
        product_type
        for product_type in PRODUCT_TYPES
        for _ in range(3)
    ]

    selected_types = random.sample(available_types, k=9)

    counters = Counter()
    table_objects = []

    for product_type in selected_types:
        counters[product_type] += 1
        table_objects.append(f"{product_type}_{counters[product_type]}")

    random.shuffle(table_objects)

    return table_objects

PRODUCT_TYPES = [
    "electronics",
    "drinks",
    "snacks",
    "hygiene",
    "textile",
]

TRASHCAN_PLACEMENT_THRESHOLD = 0.15

PRODUCT_INSTANCES = [
    f"{product_type}_{i}"
    for product_type in PRODUCT_TYPES
    for i in range(8)
]

KNOWN_ROBOTS = [
    "reception_robot",
    "preparation_robot_a",
    "preparation_robot_b",
]

MOVE_TARGETS = [
    "reception",
    "electronics_spot",
    "drinks_spot",
    "snacks_spot",
    "hygiene_spot",
    "textile_spot",
    "priority_bay_0",
    "priority_bay_1",
    "standard_bay_0",
    "standard_bay_1",
]


att = {
    "known_robots": KNOWN_ROBOTS,
    "product_types": PRODUCT_TYPES,
    "move_targets": MOVE_TARGETS
}

MAIN_COURSE = ["chicken", "fish", "pasta"]
FRUITS = ["ananas", "apple", "banana"]
DRINKS = ["water", "juice", "milk"]

FOOD_BY_TYPE = {
    "main_course": MAIN_COURSE,
    "fruits": FRUITS,
    "drinks": DRINKS,
}

FOOD_TYPES = list(FOOD_BY_TYPE)
FOOD_OBJECTS = [
    food
    for foods in FOOD_BY_TYPE.values()
    for food in foods
]
OBJECT_TYPE = {
    food: food_type
    for food_type, foods in FOOD_BY_TYPE.items()
    for food in foods
}

FOOD_TYPE_SINGULAR = {
    "main_course": "main course",
    "fruits": "fruit",
    "drinks": "drink",
}

# Backward-compatible names used by the environment, tools and old benchmark.
main_course = MAIN_COURSE
fruits = FRUITS
drinks = DRINKS

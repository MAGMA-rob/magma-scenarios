import random

CLEAN_STATE = 1
DIRTY_STATE = 0
STATIC_STATE = -1

food = ["banana_1","banana_2","apple_1","apple_2"]

dishware = ["plate_1","plate_2","glass_1","glass_2","cup_1","cup_2"]

cleaning_objects = ["sponge"]

movable_objects = [*food, *dishware, *cleaning_objects]

food_types = list(
    dict.fromkeys(object_name.rsplit("_", 1)[0] for object_name in food)
)
dishware_types = list(
    dict.fromkeys(object_name.rsplit("_", 1)[0] for object_name in dishware)
)
cleaning_object_types = cleaning_objects.copy()

COMMON_LOCATIONS = ["table","food_storage","dish_storage","trashcan"]

DISHWASHER_LOCATIONS = [*COMMON_LOCATIONS,"washing_machine",]

ADVANCED_LOCATIONS = [*COMMON_LOCATIONS,"sink","drying_zone"]

object_classes = ["food","dishware","cleaning_objects"]

att = {
    "food": food,
    "dishware": dishware,
    "objects": cleaning_objects,
}

generic_rules = [
    "Dirty dishware must be cleaned in the sink and dirty food must be placed in the trashcan.",
    "dish_robot can only take dishware from the table, sink, or drying_zone.",
    "Both robots can take from and put clean dishware in drying_zone.",
    "Only table_robot can access food_storage and dish_storage.",
]
generic_rules_simplified = [
    "Dirty food cannot be cleaned or stored and must be placed in the trashcan.",
    "Clean food must be placed in food_storage.",
    "Clean dishware must be placed in dish_storage."
]

def build_task_attributes(locations):
    return {
        "locations": locations,
        "object_classes": object_classes,
    }


def build_complete_task_attributes(
    attributes: dict[str, list[str]],
) -> dict[str, list[str]]:
    return {
        **attributes,
        "food_types": food_types.copy(),
        "dishware_types": dishware_types.copy(),
        "cleaning_object_types": cleaning_object_types.copy(),
    }


def build_random_env_start(max_on_table: int = 4):
    selected_food = random.sample(food, k=3)
    selected_dishware = random.sample(dishware, k=3)
    selected_objects = selected_food + selected_dishware

    if max_on_table <= 0:
        table = []
    else:
        count = random.randint(
            0,
            min(max_on_table, len(selected_objects)),
        )
        table = random.sample(selected_objects, k=count)

    storage = [obj for obj in selected_objects if obj not in table]

    dirty = [obj for obj in table if random.random() < 0.5]

    return table, storage, dirty

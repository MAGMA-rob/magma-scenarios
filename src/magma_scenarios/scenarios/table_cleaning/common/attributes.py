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
    "Cleaning the table means removing all food and dishware. "
    "Setting it means leaving only the requested types and counts of clean objects: "
    "reuse matching clean objects already there, and remove extras or dirty objects.",
    "Use scan_scene to find objects, their locations, and which ones are dirty.",
    "The table_robot puts dirty food in trashcan and clean food in food_storage.",
    "For dirty dishware, the dish_robot puts it in sink and uses clean; clean moves it to drying_zone.",
    "The table_robot takes clean dishware from drying_zone and puts it in dish_storage.",
    "Only the table_robot can access food_storage and dish_storage.",
]
generic_rules_simplified = [
    "Cleaning the table means removing all food and dishware from it; wipe it when requested.",
    "Take an object and use inspect to determine whether it is clean or dirty before choosing its destination.",
    "Put dirty food in trashcan and clean food in food_storage.",
    "Put dirty dishware in washing_machine and clean dishware in dish_storage.",
    "To wipe the table, first clear it, take the sponge, use wipe, then put the sponge back on the table.",
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

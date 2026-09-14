import random


ROBOT_NAMES = ["robot1", "robot2"]

HALLS = ["Hall1", "Hall2", "Hall3", "Hall4"]

OBJECT_TYPES = ["book", "pen", "backpack", "package"]

OBJECT_NAMES = [f"{obj_type}_{idx}" for obj_type in OBJECT_TYPES for idx in range(1, 5)]

def build_random_env_start(max_per_zone: int = 4):
    
    all_objects = OBJECT_NAMES.copy()
    random.shuffle(all_objects)

    env_option = {hall : [] for hall in HALLS}

    for obj in all_objects:
        available_hall = [
            hall_name
            for hall_name, objects in env_option.items()
            if len(objects) < max_per_zone
        ]

        if not available_hall:
            break

        hall_name = random.choice(available_hall)
        env_option[hall_name].append(obj)
    
    return env_option

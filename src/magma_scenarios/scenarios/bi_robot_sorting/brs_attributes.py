import random
import copy 

ZONES = ['left_zone', 'mutual_zone', 'right_zone']

ROBOTS = ["arm1", "arm2"]

OBJECT_TYPES = ["banana", "apple", "orange"]

sorting_objects = {
    "banana" :[ "banana_0", "banana_1", "banana_2"],
    "apple" : ["apple_0", "apple_1", "apple_2"],
    "orange" : ["orange_0", "orange_1", "orange_2"],
}



def build_random_env_start(
    max_per_zone: int = 4,
    min_damaged_objects: int = 1,
    max_damaged_objects: int = 3,
):
    all_objects = [obj for obj_type in sorting_objects.values() for obj in obj_type]

    if not 0 <= min_damaged_objects <= max_damaged_objects <= len(all_objects):
        raise ValueError(
            "Damaged object limits must fit within the available objects."
        )

    random.shuffle(all_objects)

    zones = {
        "left_zone": [],
        "mutual_zone": [],
        "right_zone": [],
    }

    for obj in all_objects:
        available_zones = [
            zone_name
            for zone_name, objects in zones.items()
            if len(objects) < max_per_zone
        ]

        if not available_zones:
            break

        zone_name = random.choice(available_zones)
        zones[zone_name].append(obj)

    damaged_count = random.randint(min_damaged_objects, max_damaged_objects)
    dirty_obj = random.sample(all_objects, k=damaged_count)
    
    return zones["left_zone"], zones["mutual_zone"], zones["right_zone"], dirty_obj

def perturb_assignment(assignment: dict[str, dict[str, int]], max_moves: int) -> tuple[dict[str, dict[str, int]], int]:
    if max_moves < 1:
        raise ValueError(
            f"max_nb_misplaced must be at least 1, got {max_moves}"
        )

    perturbed = copy.deepcopy(assignment)

    actual_misplaced = 0
    increased = set()
    decreased = set()

    for _ in range(max_moves):
        possible_moves = []

        for from_zone in ZONES:
            for obj_type in OBJECT_TYPES:
                if perturbed[from_zone][obj_type] <= 0:
                    continue

                if (from_zone, obj_type) in increased:
                    continue

                for to_zone in ZONES:
                    if to_zone == from_zone:
                        continue

                    if (to_zone, obj_type) in decreased:
                        continue

                    possible_moves.append((from_zone, to_zone, obj_type))

        if not possible_moves:
            break

        from_zone, to_zone, obj_type = random.choice(possible_moves)

        perturbed[from_zone][obj_type] -= 1
        perturbed[to_zone][obj_type] += 1

        decreased.add((from_zone, obj_type))
        increased.add((to_zone, obj_type))

        actual_misplaced += 1

    return perturbed, actual_misplaced

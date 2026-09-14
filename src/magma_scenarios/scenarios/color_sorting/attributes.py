from typing import Any, Dict, List, Tuple
import random 
import copy 

available_colors = ["green","yellow","black","white","blue","red"]

def _empty_assignment(colors: List[str]) -> Dict[str, Dict[str, int]]:
    assignment = {
        "table": {color: 0 for color in colors},
    }

    for color in colors:
        assignment[f"{color}_tray"] = {
            other_color: 0
            for other_color in colors
        }

    return assignment


def _build_goal_assignment(colors: List[str], cubes_per_color: int) -> Dict[str, Dict[str, int]]:
    assignment = _empty_assignment(colors)

    for color in colors:
        assignment[f"{color}_tray"][color] = cubes_per_color

    return assignment


def _perturb_assignment(assignment: Dict[str, Dict[str, int]], max_misplaced: int = 5) -> Tuple[Dict[str, Dict[str, int]], int]:
    if max_misplaced < 0:
        raise ValueError(f"max_misplaced must be >= 0, got {max_misplaced}")

    perturbed = copy.deepcopy(assignment)
    actual_misplaced = 0

    locations = list(assignment.keys())

    object_types = sorted({
        object_type
        for type_counts in assignment.values()
        for object_type in type_counts.keys()
    })

    increased = set()
    decreased = set()

    for _ in range(max_misplaced):
        possible_moves = []

        for source in locations:
            for object_type in object_types:
                if perturbed[source].get(object_type, 0) <= 0:
                    continue

                if (source, object_type) in increased:
                    continue

                for destination in locations:
                    if destination == source:
                        continue

                    if (destination, object_type) in decreased:
                        continue

                    possible_moves.append((source, destination, object_type))

        if not possible_moves:
            break

        source, destination, object_type = random.choice(possible_moves)

        perturbed[source][object_type] -= 1
        perturbed[destination][object_type] += 1

        decreased.add((source, object_type))
        increased.add((destination, object_type))

        actual_misplaced += 1

    return perturbed, actual_misplaced
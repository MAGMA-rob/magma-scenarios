from typing import Dict, List
import random
from collections import defaultdict

from magma_core.base.stage import BaseTaskStage
from magma_core.base.user_request import BaseRequest, BaseConstraintRequest
from magma_core.base.state import TaskState
from magma_core.base.data_structures import UserInstruction

from .coffee_stages import MakeOneCoffeStage, CoffeeCompositeStage

from magma_scenarios.templates.stages import MissingInformationStage
from magma_scenarios.templates.constraints import RelationAssignmentConstraint

class AskCoffeeRequest(BaseRequest):

    def __init__(self, max_coffee_making : int = 2, strict_order : bool = True):
        super().__init__()
        self.max_nb = max_coffee_making
        self.strict_order = strict_order
        if strict_order == False:
            raise ValueError("Strict order to True is not currently supported")

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:

        pods = state.attributes.get("coffee_pod",[])
        n = random.randint(1,self.max_nb)

        if len(pods) == 0:
            raise RuntimeError("Failed to find pods")

        pods_seq = random.choices(pods,k=n)
        stages = []
        coffee_str = ' and '.join(pods_seq)
        instruction = f"Hello, please make thse coffee in this exact order: {coffee_str}"
        for i in range(n):
            stages.append(MakeOneCoffeStage(
                pods_seq[i],
                instruction,
                [],
                flag_answer=i==n-1
            ))
            instruction="none"

        return stages

class GiveCoffeePreference(BaseConstraintRequest):

    def __init__(self, possible_names : List[str], max_name : int = 4, max_different_name : int = 6):
        super().__init__()
        self.max_nb = max_name
        self.possible_names = possible_names
        self.max_difference = max_different_name

    def initialize_constraints(self, state: TaskState):
        self.constraints = []

        coffee = state.attributes.get("coffee_pod",[])
        if len(coffee) <= 0:
            raise RuntimeError("Empty coffee pod")
        
        # If we are at the maximum of different preference we just modify existing ones
        if len(state.relations.get("coffee_preference",{})) >= self.max_difference:
            n = random.randint(1,min(len(self.possible_names),self.max_nb))
            names = random.sample(list(state.relations.get("coffee_preference",{}).keys()),k=n)
        else:
            # Otherwise we sample new ones
            max_diff = self.max_difference - len(state.relations.get("coffee_preference",{}))
            n = random.randint(1,min(len(self.possible_names),self.max_nb,max_diff))
            names = random.sample(self.possible_names,k=n)

        coffees = random.choices(coffee,k=n)
        
        assignment = defaultdict(list)
        for name, pod in zip(names,coffees):
            assignment[pod].append(name)
            self.constraints.append(
                CoffeePreferenceConstraints(
                    name,pod
                )
            )
        self.constraint_msg = "Hey, please remember that "
        for i, (pod, name_list) in enumerate(assignment.items()):
            names_str = " and ".join(name_list)
            self.constraint_msg += names_str + " like their coffee " + str(pod)
            if i != len(assignment)-1:
                self.constraint_msg += ", "
        self.constraint_msg += "."

class AskCoffeePerUser(BaseRequest):

    def __init__(
            self,
            possible_names : List[str],
            max_coffee : int = 3,
            force_order : bool = True
        ) -> None:
        super().__init__()
        self.nb = max_coffee
        self.possible_names = possible_names
        self.force_order = force_order

    def _join_names(self, names: List[str]) -> str:
        if len(names) == 1:
            return names[0]
        return " and ".join(names)

    def _build_missing_preference_answer(self, missing_names: List[str]) -> str:
        joined_names = self._join_names(missing_names)
        if len(missing_names) == 1:
            return f"The model must inform that {joined_names} does not have any coffee preference"
        return f"The model must inform that {joined_names} do not have any coffee preference"

    def _build_preference_resolution(self, missing_assignment: Dict[str, str]) -> str:
        if len(missing_assignment) == 0:
            return ""

        parts = [f"{name} wants a {pod} coffee" for name, pod in missing_assignment.items()]
        if len(parts) == 1:
            return parts[0] + "."
        if len(parts) == 2:
            return " and ".join(parts) + "."
        return ", ".join(parts[:-1]) + f", and {parts[-1]}."

    def _build_ordered_instruction(self, names: List[str]) -> str:
        if len(names) == 1:
            return f"Please make the coffee for {names[0]}."

        parts = [f"first the coffee for {names[0]}"]
        for name in names[1:]:
            parts.append(f"then the one for {name}")

        if len(parts) == 2:
            ordered_names = " and ".join(parts)
        else:
            ordered_names = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        return f"Please make {ordered_names}."

    def _sample_requested_users(self, state: TaskState) -> tuple[List[str], Dict[str, str]]:
        pods = state.attributes.get("coffee_pod", [])
        if len(pods) == 0:
            raise RuntimeError("Empty coffee pod")

        preference: Dict[str, str] = state.relations.get("coffee_preference", {})
        known_names = list(preference.keys())
        random.shuffle(known_names)

        all_names = known_names.copy()
        for name in self.possible_names:
            if name not in preference and name not in all_names:
                all_names.append(name)

        if len(all_names) == 0:
            raise RuntimeError("Failed to sample a user to serve coffee")

        n = random.randint(1, min(self.nb, len(all_names)))

        selected_names = known_names[:min(n, len(known_names))]
        if len(selected_names) < n:
            missing_names = [name for name in all_names if name not in preference]
            random.shuffle(missing_names)
            selected_names.extend(missing_names[:n - len(selected_names)])

        missing_assignment = {
            name: random.choice(pods)
            for name in selected_names
            if name not in preference
        }
        return selected_names, missing_assignment

    def _get_capsule(
            self,
            name: str,
            preference: Dict[str, str],
            missing_assignment: Dict[str, str]
        ) -> str:
        if name in missing_assignment:
            return missing_assignment[name]
        return preference[name]

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        stages = []

        preference: Dict[str, str] = state.relations.get("coffee_preference", {})
        names, missing_assignment = self._sample_requested_users(state)

        request_instruction = f"Please make coffee for {self._join_names(names)}."
        if missing_assignment:
            stages.append(MissingInformationStage(
                UserInstruction(request_instruction),
                self._build_missing_preference_answer(list(missing_assignment.keys())),
                state.memory,
                state.attributes
            ))

        ordered_instruction = self._build_ordered_instruction(names)
        resolution_instruction = self._build_preference_resolution(missing_assignment)

        if self.force_order:
            first_instruction = ordered_instruction
            if resolution_instruction:
                first_instruction = f"{resolution_instruction} {ordered_instruction}"

            for i, name in enumerate(names):
                capsule = self._get_capsule(name, preference, missing_assignment)
                stages.append(MakeOneCoffeStage(
                    capsule,
                    first_instruction if i == 0 else "none",
                    [],
                    i == len(names) - 1
                ))
            return stages

        if len(names) == 1:
            name = names[0]
            capsule = self._get_capsule(name, preference, missing_assignment)
            instruction = resolution_instruction if resolution_instruction else request_instruction
            stages.append(MakeOneCoffeStage(capsule, instruction, [], True))
            return stages

        number = defaultdict(int)
        for name in names:
            number[self._get_capsule(name, preference, missing_assignment)] += 1
        stages.append(CoffeeCompositeStage(dict(number)))

        return stages

### CONSTRAINTS

class CoffeePreferenceConstraints(RelationAssignmentConstraint):
    
    def __init__(
            self,
            name: str,
            coffee: str
        ) -> None:
        super().__init__(name, coffee, "coffee_preference", None, "coffee_pod")



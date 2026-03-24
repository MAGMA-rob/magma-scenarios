from typing import Dict, Any, List
import random
from collections import defaultdict
from copy import deepcopy

from magma_core.base.stage import BaseTaskStage, ModifAttributesBaseStage
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
        self.max_difference = self.max_difference

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

    def __init__(self,  possible_names : List[str], max_coffee : int = 3) -> None:
        super().__init__()
        self.nb = max_coffee
        self.possible_names = possible_names

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        stages = []

        preference : Dict = state.relations.get("coffee_preference",{})
        names = list(preference.keys())
        new_names = [] # not known yet names
        random.shuffle(names)

        n = random.randint(1,self.nb)

        if n > len(names):
            random.shuffle(self.possible_names)
            for name in self.possible_names:
                if n >= len(names):
                    break
                if name not in names:
                    new_names.append(name)
                    n+=1

        
        missing_assignment = {}
        for name in new_names:
            missing_assignment[name] = random.choice(state.attributes.get("coffee_pod",[]))

        instruction = f"Please make coffee for : {' and '.join(names+new_names)}"
        if new_names:
            MissingInformationStage(
                UserInstruction(instruction),
                f"The model must inform that {' and '.join(new_names)} does not have any coffee preference",
                state.memory, state.attributes 
            )
            instruction = ""
            for i, name in enumerate(new_names):
                instruction += f"{name} want a {missing_assignment[name]} coffee"
                if i != len(new_names)-1:
                    instruction += ", "

        if n == 1:
            if new_names:
                capsule = missing_assignment[new_names[0]]
            else:
                capsule = preference[names[0]]
            stages.append(MakeOneCoffeStage(capsule,instruction,[],True))
        else:
            number = {k:0 for k in state.attributes.get("coffee_pod",[])}
            for name in names:
                number[preference[name]] += 1 
            for name, pod in missing_assignment.items():
                number[pod] += 1
            stages.append(CoffeeCompositeStage(number))

        return stages

### CONSTRAINTS

class CoffeePreferenceConstraints(RelationAssignmentConstraint):
    
    def __init__(
            self,
            name: str,
            coffee: str
        ) -> None:
        super().__init__(name, coffee, "coffee_preference", None, "coffee_pod")
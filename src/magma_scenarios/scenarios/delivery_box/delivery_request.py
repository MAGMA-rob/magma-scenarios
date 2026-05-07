import random
from typing import Dict, List, Optional

from magma_core.base.stage import BaseTaskStage
from magma_core.base.state.task_state import TaskState
from magma_core.base.user_request import BaseConstraintRequest, BaseRequest
from magma_core.base.constraints import BaseConstraint
from magma_core.utils.env_utils import craft_random_manu_order

from .attributes import MAX_NB_PER_RECIPE
from .delivery_stages import CycleStage

RECIPE_NEEDS_APPLICATION_KEY = "recipe_needs_application"
RECIPE_OVERRIDE_NEEDS_APPLICATION_KEY = "recipe_override_needs_application"
RECIPE_APPLICATIONS_KEY = "recipe_applications"


def build_recipe_instruction(products: List[str], action: Optional[str] = None) -> str:
    recipe_counts: Dict[str, int] = {}
    for product_name in products:
        recipe_counts[product_name] = recipe_counts.get(product_name, 0) + 1

    instruction = " and ".join(
        f"{count} {product_name}" for product_name, count in recipe_counts.items()
    )

    if action is None or len(instruction) == 0:
        return instruction

    return f"{action} {instruction}"


class RecipeConstraints(BaseConstraint):

    def __init__(self, add : List[str], remove : List[str], overridde : bool = False) -> None:
        super().__init__()
        self.to_add = add
        self.to_remove = remove
        self.overridde = overridde

    def apply(self, state: TaskState):
        super().apply(state)
        if self.overridde:
            state.properties["recipe"] = self.to_add.copy()
            state.properties[RECIPE_OVERRIDE_NEEDS_APPLICATION_KEY] = True
        else:
            for r in self.to_remove:
                state.properties["recipe"].remove(r) # Will raise an exception if not valid
            state.properties["recipe"].extend(self.to_add.copy())
        state.properties[RECIPE_NEEDS_APPLICATION_KEY] = True
    

class GiveRecipe(BaseConstraintRequest):

    def __init__(self, max_recipe_lenght : int = 3):
        super().__init__()
        self.max_lenght = max_recipe_lenght

    def sampling_weight(self, state: TaskState) -> float:
        if state.properties.get(RECIPE_OVERRIDE_NEEDS_APPLICATION_KEY, False):
            return 0
        if len(state.attributes.get("product_type", [])) <= 0:
            return 0
        if len(state.properties.get("recipe", [])) <= 0:
            return 4
        if state.properties.get(RECIPE_NEEDS_APPLICATION_KEY, False):
            return 0.5
        return 1

    def initialize_constraints(self, state: TaskState):
        all_type = 2*state.attributes.get("product_type",[]).copy()
        random.shuffle(all_type)

        n = random.randint(1,self.max_lenght)
        recipe = all_type[:n]

        self.constraints = [RecipeConstraints(add=recipe,remove=[],overridde=True)]
        self.constraint_msg = f"Please update the default recipe to: {build_recipe_instruction(recipe)}."

class UpdateRecipe(BaseConstraintRequest):
    
    def __init__(self, max_update : int = 2):
        super().__init__()
        self.max_lenght = max_update

    def sampling_weight(self, state: TaskState) -> float:
        cur_recipe = state.properties.get("recipe", [])
        product_types = state.attributes.get("product_type", [])
        can_add = any(
            cur_recipe.count(product_name) < MAX_NB_PER_RECIPE
            for product_name in product_types
        )
        can_remove = len(cur_recipe) > 1
        return 1 if can_add or can_remove else 0

    def initialize_constraints(self, state: TaskState):
        cur_recipe : List[str] = state.properties.get("recipe",[]).copy()
        possible = []
        for p in state.attributes['product_type']:
            n = cur_recipe.count(p)
            possible.extend([p]*(MAX_NB_PER_RECIPE-n))

        n = random.randint(1,self.max_lenght)
        to_add = []
        to_remove = []
        for _ in range(n):
            can_add = len(possible) > 0
            can_remove = len(cur_recipe) > 1
            if not can_add and not can_remove:
                break
            if not can_remove:
                ac = 1
            elif not can_add:
                ac = 0
            else:   
                ac = random.randint(0,1)
            
            if ac == 1:
                select = random.choice(possible)
                possible.remove(select)
                to_add.append(select)
                cur_recipe.append(select)
            else:
                select = random.choice(cur_recipe)
                cur_recipe.remove(select)
                to_remove.append(select)
                possible.append(select)

        self.constraints = [RecipeConstraints(add=to_add,remove=to_remove)]
        actions = []
        if len(to_add) > 0:
            actions.append(f"{build_recipe_instruction(to_add, 'add')} to the recipe")
        if len(to_remove) > 0:
            actions.append(f"{build_recipe_instruction(to_remove, 'remove')} from the recipe")
        self.constraint_msg = f"Hello! I want you to {' and '.join(actions)}."


class AskForCycle(BaseRequest):

    def __init__(self) -> None:
        super().__init__()

    def sampling_weight(self, state: TaskState) -> float:
        if len(state.properties.get("recipe",[])) <= 0:
            return 0
        if state.properties.get(RECIPE_NEEDS_APPLICATION_KEY, False):
            return 8
        return 2

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        recipe = state.properties.get("recipe",[])
        if len(recipe) <= 0:
            raise RuntimeError("Empty recipe has reached the create stage")
        manu = craft_random_manu_order(3)
        nb = random.randint(1,99)
        return [
            CycleStage(recipe,manu,nb,f"Hello, I need {nb} deliveries under manufacturing order {manu}",True)
        ]

    def apply_request(self, state: TaskState) -> TaskState:
        if state.properties.get(RECIPE_NEEDS_APPLICATION_KEY, False):
            state.properties[RECIPE_NEEDS_APPLICATION_KEY] = False
            state.properties[RECIPE_OVERRIDE_NEEDS_APPLICATION_KEY] = False
            state.properties[RECIPE_APPLICATIONS_KEY] = (
                state.properties.get(RECIPE_APPLICATIONS_KEY, 0) + 1
            )
        return state
    
class AskForCycleWithOverride(BaseRequest):

    def __init__(self) -> None:
        super().__init__()

    def sampling_weight(self, state: TaskState) -> float:
        if len(state.attributes.get("product_type", [])) <= 0:
            return 0
        if state.properties.get(RECIPE_NEEDS_APPLICATION_KEY, False):
            return 0.25
        return 1

    def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
        all_types = state.attributes.get("product_type",[]).copy()
        random.shuffle(all_types)
        recipe = []
        manu = craft_random_manu_order(3)
        nb = random.randint(1,99)

        for product_name in all_types:
            n = random.randint(0, MAX_NB_PER_RECIPE)
            if n > 0:
                recipe.extend(n * [product_name])

        if len(recipe) == 0 and len(all_types) > 0:
            recipe.append(random.choice(all_types))

        instruction = (
            f"Hey, Can you do me {nb} packs under order {manu} "
            f"with: {build_recipe_instruction(recipe)}."
        )
        
        return [
            CycleStage(recipe,manu,nb,instruction,True)
        ]
    
# class MultipleCycleInARow(BaseRequest):

#     def __init__(self, max_nb : int = 3) -> None:
#         super().__init__()
#         self.max_nb = max_nb

#     def create_stages(self, state: TaskState) -> List[BaseTaskStage]:
#         nb_stage = random.randint(2,self.max_nb)
#         manu_order = craft_random_manu_order(2)
#         all_types = state.properties.get("product_type",[]).copy()
#         random.shuffle(all_types)

#         for _ in range(nb_stage):


#         return super().create_stages(state)

from magma_core.base.tasks import BaseBenchmarkTask
from magma_scenarios.templates.errors.perception_errors import MaskedObjectError
from .load_tools import CookingTool
from .attributes import main_course, desserts, drinks
from .cooking_errors import GraspFoodFailureError, MaskRemainingFoodError

class cookingBenchmarks(BaseBenchmarkTask) :
    name: str = "Benshmark cooking"
    env_id: str = "Cooking"

    Tools_cls = CookingTool

    env_options = {}
    all_task_attributes = {
        "main_course" : main_course,
        "desserts" : desserts,
        "drinks" : drinks
    }

    benchmark_possible_errors = [GraspFoodFailureError(2),MaskRemainingFoodError(2)]
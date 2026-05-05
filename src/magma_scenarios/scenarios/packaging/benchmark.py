from magma_core.base.tasks import BaseBenchmarkTask
from magma_scenarios.templates.errors.perception_errors import MaskedObjectError
from .packaging_tools import PackagingTool
from .attributes import main_course, fruits, drinks
from .packaging_errors import GraspFoodFailureError, MaskFoodError

class PackagingBenchmarks(BaseBenchmarkTask) :
    name: str = "Benshmark packaging"
    env_id: str = "Packaging"

    Tools_cls = PackagingTool

    env_options = {}
    all_task_attributes = {
        "main_course" : main_course,
        "fruits" : fruits,
        "drinks" : drinks
    }

    benchmark_possible_errors = [GraspFoodFailureError(2),MaskFoodError(2)]
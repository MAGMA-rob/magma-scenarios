from pathlib import Path
from magma_core.base.tasks import BaseBenchmarkTask
from .load_tool import LaunchTool
from .attributes import all_clothes, all_detergents
from .laundry_errors import GraspClothesFailureError

class LaundryBenchmark(BaseBenchmarkTask):
    """
    Benchmark for Laundry scenario.
    Tests constraint following, multi-step reasoning, and team-based preference rules.
    """

    name: str = "Benchmark  Laundry"
    env_id: str = "Laundry-v1"  

    Tools_cls = LaunchTool 


    env_options = {}

    all_task_attributes = {
        "all_clothes": all_clothes,
        "all_detergent": all_detergents
    }

    benchmark_possible_errors = [
        GraspClothesFailureError(5)
    ]
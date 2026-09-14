from pathlib import Path
from magma_core.simulation.tasks import BaseBenchmarkTask
from .load_tool import LaunchTool
from .attributes import all_clothes, all_detergents
from .laundry_errors import GraspClothesFailureError

RANDOMIZED_CONFIG_PATH = str(Path(__file__).resolve().parent / "laundry.yaml")

class LaundryBenchmark(BaseBenchmarkTask):
    """
    Benchmark for Laundry scenario.
    Tests constraint following, multi-step reasoning, and team-based preference rules.
    """

    name: str = "Benchmark  Laundry"
    env_id: str = "Laundry-v1"  
    randomized_config_path = RANDOMIZED_CONFIG_PATH

    Tools_cls = LaunchTool 


    env_options = {}

    all_task_attributes = {
        "all_clothes": all_clothes,
        "all_detergent": all_detergents
    }

    benchmark_possible_errors = [
        GraspClothesFailureError()
    ]

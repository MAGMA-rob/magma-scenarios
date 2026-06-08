from pathlib import Path
import sapien
from .coffee_errors import GraspCapsuleFailureError
from magma_core.base.tasks import BaseBenchmarkTask
from .simple_tool import MakingCoffeeTool
from .attributes import att, people, teams, loaded_capsule_pose, dropped_mug_pose
import sapien

RANDOMIZED_CONFIG_PATH = str(Path(__file__).resolve().parent / "make_coffee_cfg.yaml")

class CoffeeBenchmark(BaseBenchmarkTask):
    """
    Benchmark for the coffee making scenario.
    Tests constraint following, multi-step reasoning, and team-based preference rules.
    """

    name: str = "Benchmark Coffee Making"
    env_id: str = "MakeCoffee-v1"  
    randomized_config_path = RANDOMIZED_CONFIG_PATH

    Tools_cls = MakingCoffeeTool 

    env_options = {}

    tools_constant = {
        "teams": {
            "DISCO" : ["Smith", "Anderson", "Clark", "Wright"],
            "GEPETTO" : ["Mitchell", "Johnson", "Thomas", "Rodriguez"],
            "RAP" : ["Lopez", "Perez", "Williams", "Jackson"],
            "RIS" : ["Lewis", "Hill", "Roberts", "Jones"],
            "MAC" : ["White", "Lee", "Scott", "Turner"],
        },
        "loaded_capsule_pose": loaded_capsule_pose,
        "dropped_mug_pose" : dropped_mug_pose,
        "base_pose" : sapien.Pose(p = [-0.1,0,0.4],q = [0,1,0,0])
    }

    all_task_attributes = att

    benchmark_possible_errors = [GraspCapsuleFailureError(2)]

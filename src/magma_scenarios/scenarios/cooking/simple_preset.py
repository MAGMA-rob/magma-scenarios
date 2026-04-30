from magma_core.base.tasks import BaseTask
from magma_core.base.tasks_style import TaskStyle
from .simple_stage import SimplePutOnTrayStage
from magma_scenarios.scenarios.cooking.load_tools import CookingTool
import sapien


class SimpleCookingPreset(BaseTask):
    """
    Minimal preset for tool testing
    """

    env_id = "Cooking"
    name = "SimpleCookingPreset"
    Tools_cls = CookingTool
    styles = [ TaskStyle.CONSTRAINED]
    tools_constant = {
        "base_pose": sapien.Pose(p=[0, 0, 0.4], q=[0, 1, 0, 0])
    }

    all_task_attributes = {}

    def __init__(self):
        super().__init__()

        self.stages = [
            SimplePutOnTrayStage()
        ]
        self.approximal_difficulty = "Medium"
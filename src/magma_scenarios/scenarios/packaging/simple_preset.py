from pathlib import Path

import sapien

from magma_core.simulation.data_structures import SituationInit
from magma_core.simulation.tasks import BaseTask, TaskMetadata
from magma_core.simulation.tasks_style import TaskStyle

from magma_scenarios.scenarios.packaging.packaging_tools import PackagingTool

from .attributes import drinks, fruits, main_course
from .simple_stage import SimplePutOnTrayStage


class SimplePackagingPreset(BaseTask):
    """
    Minimal preset for tool testing
    """

    maniskill_env_id = "Packaging"
    name = "SimplePackagingPreset"
    Tools_cls = PackagingTool
    randomized_config_path = str(Path(__file__).resolve().parent / "packaging.yaml")
    tools_constant = {
        "base_pose": sapien.Pose(p=[0, 0, 0.4], q=[0, 1, 0, 0])
    }

    def __init__(self):
        super().__init__()

        self.situation_init = SituationInit(
            attributes={
                "main_course": main_course.copy(),
                "fruits": fruits.copy(),
                "drinks": drinks.copy(),
            }
        )
        self.task_metadata = TaskMetadata(
            styles=[TaskStyle.CONSTRAINED],
            approximal_difficulty="Medium",
        )
        self.stages = [SimplePutOnTrayStage()]

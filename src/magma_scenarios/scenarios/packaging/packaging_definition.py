from copy import deepcopy
from pathlib import Path

import sapien

from magma_core.simulation.data_structures import SituationInit
from magma_core.simulation.state import TaskState
from magma_core.simulation.tasks import (
    InitializationParameters,
    TaskDefinition,
    TaskMetadata,
)
from magma_core.simulation.tasks_style import TaskStyle

from .attributes import FOOD_BY_TYPE, OBJECT_TYPE
from .packaging_constraints import (
    DEFAULT_RECIPE_KEY,
    DEFAULT_RECIPE_PENDING_KEY,
    INCOMPATIBILITY_RULES_KEY,
    INCOMPATIBILITY_RULES_PENDING_KEY,
)
from .packaging_tools import PackagingTool
from .requests import (
    PackagingInterruptionRequest,
    PrepareTrayRequest,
    ReplaceTrayItemRequest,
    UpdateDefaultRecipeRequest,
    UpdateIncompatibilityRuleRequest,
)
from .rule_renderer import PackagingRuleRenderer


RANDOMIZED_CONFIG_PATH = str(Path(__file__).resolve().parent / "packaging.yaml")


class PackagingDefinition(TaskDefinition):
    maniskill_env_id = "Packaging"
    Tools_cls = PackagingTool
    RuleRenderer_cls = PackagingRuleRenderer

    def __init__(self) -> None:
        attributes = {
            food_type: foods.copy()
            for food_type, foods in FOOD_BY_TYPE.items()
        }
        attributes["known_robots"] = ["panda"]

        super().__init__(
            name="Tray packaging with recipes and incompatibility rules",
            situation_init=SituationInit(
                attributes=deepcopy(attributes),
                all_task_attributes=deepcopy(attributes),
                memory={
                    "memory_list": [
                        "To prepare a tray, take each required food object "
                        "and put it on the tray."
                    ]
                },
            ),
            initialization_parameters=InitializationParameters(
                agent_names=["panda"],
            ),
            randomized_config_path=RANDOMIZED_CONFIG_PATH,
            task_metadata=TaskMetadata(
                styles=[TaskStyle.CONSTRAINED],
                approximal_difficulty="Medium",
            ),
            tools_constant={
                "base_pose": sapien.Pose(p=[0, 0, 0.4], q=[0, 1, 0, 0])
            },
            active_requests=[
                UpdateDefaultRecipeRequest(),
                UpdateIncompatibilityRuleRequest(),
                PrepareTrayRequest(),
                ReplaceTrayItemRequest(),
                PackagingInterruptionRequest(),
            ],
        )

        self.starting_state = TaskState()
        self.starting_state.attributes = deepcopy(attributes)
        self.starting_state.memory = deepcopy(self.situation_init.memory)
        self.starting_state.relations = {
            "object_type": OBJECT_TYPE.copy(),
        }
        self.starting_state.properties = {
            DEFAULT_RECIPE_KEY: [],
            INCOMPATIBILITY_RULES_KEY: [],
            DEFAULT_RECIPE_PENDING_KEY: False,
            INCOMPATIBILITY_RULES_PENDING_KEY: False,
            "packaging_rule_last_request_index": 0,
            "packaging_rule_last_step_index": 0,
        }

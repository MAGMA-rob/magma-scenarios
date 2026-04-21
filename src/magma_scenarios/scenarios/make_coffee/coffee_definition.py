# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat
# Arthur TANNEAU

from magma_core.base.tasks import TaskDefinition
from magma_core.base.state import TaskState

from .simple_tool import MakingCoffeeTool
from .attributes import att, loaded_capsule_pose, dropped_mug_pose, people
from .coffee_request import GiveCoffeePreference, AskCoffeeRequest, AskCoffeePerUser

from importlib import resources
import sapien


class SimpleDefinition(TaskDefinition):
    env_id = "MakeCoffee-v1"
    randomized_config_path = str(resources.files(__package__).joinpath("make_coffee_cfg.yaml"))
    Tools_cls = MakingCoffeeTool

    active_requests = [
        GiveCoffeePreference(people),
        AskCoffeeRequest(),
        AskCoffeePerUser(people, force_order=True)
    ]

    def __init__(self) -> None:  
        super().__init__(
            name="Simple definition",
            tools_constant= {"loaded_capsule_pose": loaded_capsule_pose, "dropped_mug_pose" : dropped_mug_pose, "base_pose" : sapien.Pose(p = [-0.1,0,0.4],q = [0,1,0,0])}
        )

        self.starting_state = TaskState()
        self.starting_state.attributes = att
        self.starting_state.relations = {"coffee_preference":{}}

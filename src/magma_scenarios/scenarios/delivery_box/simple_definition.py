from typing import Dict, List

from magma_core.base.state.task_state import TaskState
from magma_core.base.tasks import TaskDefinition

from .attributes import att

class EvolvingRecipeDefinition(TaskDefinition):

    def __init__(self):
        super().__init__(
            name="Evolving Recipe Definition",
            randomized_config_path=""
        )

        self.starting_state = TaskState()
        self.starting_state.attributes = att.copy()
        self.starting_state.properties = {"recipe":[]}
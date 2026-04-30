import random
from typing import Any, Dict, List

from magma_core.base.data_structures import Observation
from magma_scenarios.templates.errors import GraspFailureError

from .attributes import att

class GraspCapsuleFailureError(GraspFailureError):
    recovery_extra_steps = 1

    def __init__(
        self,
        max_impossible: int = 1,
        requested_coffee_pods: List[str] = [],
    ) -> None:
        super().__init__()
        self.max_nb = max_impossible
        self.requested_coffee_pods = set(requested_coffee_pods)


    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        if len(self.requested_coffee_pods) > 0:
            remaining = obs.task_attributes.get("coffee_pod", att["coffee_pod"])
        else:
            remaining = list(self.requested_coffee_pods)

        if len(remaining) <= 1 or self.max_nb <= 0:
            return {
                "inaccessible": [],
            }

        if len(remaining) == 2:
            nb = 1
        else:
            nb = random.randint(1, min(self.max_nb, len(remaining) - 1))
        return {"inaccessible": random.sample(remaining, k=nb)}

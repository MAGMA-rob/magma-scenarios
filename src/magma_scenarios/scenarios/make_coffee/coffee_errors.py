import random
from typing import Any, Dict, List, Optional

from magma_core.simulation.data_structures import Observation
from magma_scenarios.templates.errors import GraspFailureError

from .attributes import att

class GraspCapsuleFailureError(GraspFailureError):
    def __init__(
        self,
        max_impossible: int = 1,
        requested_coffee_pods: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.max_nb = max_impossible
        self.requested_coffee_pods = set(
            [] if requested_coffee_pods is None else requested_coffee_pods
        )

    def _to_spec_arguments(self) -> Dict[str, Any]:
        return {
            "max_impossible": self.max_nb,
            "requested_coffee_pods": sorted(self.requested_coffee_pods),
        }


    def initialize(self, obs: Observation, env_id: int) -> Dict[str, Any]:
        if len(self.requested_coffee_pods) > 0:
            remaining = list(self.requested_coffee_pods)
        else:
            remaining = obs.task_attributes.get("coffee_pod", att["coffee_pod"])

        if len(remaining) <= 1 or self.max_nb <= 0:
            return {
                "inaccessible": [],
            }

        if len(remaining) == 2:
            nb = 1
        else:
            nb = random.randint(1, min(self.max_nb, len(remaining) - 1))
        return {"inaccessible": random.sample(remaining, k=nb)}

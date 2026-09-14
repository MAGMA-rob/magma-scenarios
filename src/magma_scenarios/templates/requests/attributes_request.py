from typing import Tuple, Dict, Any, List, Optional
import random

from magma_core.simulation.state.task_state import TaskState
from magma_scenarios.templates.requests.interact_request import BaseAttributesModifRequest

class AddValueToListRequest(BaseAttributesModifRequest):
    """
    Base helper for requests that add values to list attributes.

    Subclasses sample an attribute snapshot, then use it both to build their
    stages and to update the latent task state after the interaction.
    """

    def __init__(self, modifiable_task_attributes: Dict[str, Any], max_update : int = 1) -> None:
        super().__init__(modifiable_task_attributes)
        self.max_update = max_update

    def sampling_weight(self, state: TaskState) -> float:
        out = super().sampling_weight(state)
        if out > 0:
            for k, vals in self.attributes.items():
                if not isinstance(state.attributes[k], List):
                    raise RuntimeError("A non-list attributes was given to the AddValueToListRequest")
                if len(state.attributes[k]) < len(vals):
                    return 5
        return 0
    
    def _get_random_key_value(self, all_keys : List, state_attributes : Dict) -> Optional[Tuple[str,str]]:
        random.shuffle(all_keys)
        for key in all_keys:
            possible_values = [x for x in self.attributes[key] if x not in state_attributes[key]]
            if len(possible_values) > 0:
                return key, random.choice(possible_values)
        return None
    
class RemoveValueToListRequest(BaseAttributesModifRequest):
    """
    Base helper for requests that remove values from list attributes.

    Subclasses sample an attribute snapshot, then use it both to build their
    stages and to update the latent task state after the interaction.
    """

    def __init__(self, modifiable_attributes_name : List, max_update : int = 1) -> None:
        super().__init__({k:None for k in modifiable_attributes_name})
        self.max_update = max_update

    def sampling_weight(self, state: TaskState) -> float:
        for k, _ in self.attributes.items():
            if not isinstance(state.attributes[k],List):
                raise RuntimeError("A non-list attributes was given to the AddValueToListRequest")
            if len(state.attributes[k]) > 0:
                return 1
        return 0
    
    def _get_random_key_value(self, all_keys : List, state_attributes : Dict) -> Optional[Tuple[str,str]]:
        random.shuffle(all_keys)
        for key in all_keys:
            if len(state_attributes[key]) > 0:
                return key, random.choice(state_attributes[key])
        return None

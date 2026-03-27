import random
from typing import Dict, Any

from magma_core.base.data_structures.tools import ToolResult
from magma_core.base.errors import BaseError

class MaskRemainingCubesError(BaseError):

    def __init__(self, max_masking = 2) -> None:
        super().__init__()
        self.max_nb = max_masking
        # je pourrai juste demander à ce que les classes enfant défine une fonction
        # qui retournerai la liste des objets possible à masquer.
        # le reste pourrait être une logique commune aux erreurs de perception

    def initialize(self) -> Dict[str, Any]:
        return {
            "masked" : None
        }

    def apply_pre_exec(self, arguments: Dict[str, Any]):
        return

    def apply_post_verif(self, tool_result: ToolResult, arguments: Dict[str, Any]):
        if not tool_result.details or len(tool_result.details) == 0:
            raise RuntimeError("The Localization Error was activated on a tool that does not return the details dict")

        remaining_objects = tool_result.details.get("table",[])
        if len(remaining_objects) <= 1:
            return

        masked = arguments.get("masked",None)

        if masked is None:
            if len(remaining_objects) == 2:
                nb = 1
            else:
                nb = random.randint(1,self.max_nb)
            
            masked = random.sample(remaining_objects,k=nb)
            arguments["masked"] = masked

        for m in masked:
            remaining_objects.remove(m)

        s = "This is the position of existing objects: "
        if len(tool_result.details['green_box']) == 0:
            s += "green_box is empty. "
        else:
            s += ",".join(tool_result.details["green_box"]) + " are in the green_box. "

        if len(tool_result.details['yellow_box']) == 0:
            s += "green_box is empty. "
        else:
            s += ",".join(tool_result.details['yellow_box']) + " are in the yellow_box. "

        if len(remaining_objects) > 0:
            s += ",".join(remaining_objects) + " are not sorted."

        tool_result.reason = s

    def get_description(self, arguments: Dict[str, Any] | None) -> str:
        if arguments is None or arguments.get("masked") is None:
            return "Mask some remaining cubes from perception."
        return f"Masked cubes from perception: {arguments['masked']}."

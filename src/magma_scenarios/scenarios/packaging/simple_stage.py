from magma_core.base.stage import ConstraintBaseStage
from magma_core.base.data_structures import Log, ToolResult
from magma_core.utils.env_utils import is_object_inside_target

## DUMMY STAGE JUSTE FOR LAUNCH THE PRESET ON TOOL TESTING
class SimplePutOnTrayStage(ConstraintBaseStage):
    """
    Goal: put at least 1 object on tray
    """

    def __init__(self):
        memory = [
            "You must place at least one object on the tray."
        ]

        super().__init__(
            constraint="at_least_one_object_on_tray",
            memory=memory,
            attributes={}
        )
        self.allowed_tools = ["detect", "take", "put", "valid_plate"]
        self.allow_tools_before_answer = True

    def verify(self, obs):
        extra = obs["extra"]

        tray_pose = extra["tray"]

        for name, obj in extra.items():
            if name in ["agent_tcp", "tray"]:
                continue

            if is_object_inside_target(
                obj,
                tray_pose,
                0.05
            ):
                return ToolResult(
                    True,
                    "At least one object is on tray",
                    logs=Log("ok")
                )

        return ToolResult(
            False,
            "No object on tray"
        )
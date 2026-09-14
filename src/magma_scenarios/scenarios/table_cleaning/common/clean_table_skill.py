# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict

from magma_core.simulation.data_structures import RobotToolStatus, ToolErrorFlag
from magma_core.simulation.skills import BaseSkill, CallTick, MessTick, SkillSpec, SkillTick


class DeplaceSkill(BaseSkill):
    spec = SkillSpec(
        name="deplace",
        description="Taking a specific object and putting it to a specific location",
        required_tools=["take", "put"],
        argument_schema={
            "object_name": {
                "type": "str",
                "description": "The object name to deplace.",
            },
            "target": {
                "type": "str",
                "description": "The target name.",
            },
        },
    )

    def __init__(self, arguments: Dict) -> None:
        super().__init__(arguments)
        self.object_name = arguments.get("object_name", "")
        self.target = arguments.get("target", "")
        self.failure_count = 0
        self.current_step = "none"

    def start(self) -> SkillTick:
        self.current_step = "grasping"
        return CallTick(
            tool_name="take",
            arguments={"name": self.object_name},
        )

    def tick(self, status: RobotToolStatus) -> SkillTick:
        if not status.result:
            self.failure_count += 1
            if self.failure_count >= 2:
                return MessTick(
                    message=(
                        "Deplace failed with message: "
                        f"{status.mess}"
                    ),
                    result=False,
                    error_flag=status.error_flag,
                )

            if self.current_step == "grasping":
                return CallTick(
                    tool_name="take",
                    arguments={"name": self.object_name},
                )

            if self.current_step == "putting":
                return CallTick(
                    tool_name="put",
                    arguments={"target": self.target},
                )

            raise RuntimeError(
                f"Unknown deplace skill step {self.current_step!r}"
            )

        if self.current_step == "grasping":
            self.failure_count = 0
            self.current_step = "putting"
            return CallTick(
                tool_name="put",
                arguments={"target": self.target},
            )

        return MessTick(
            message=f"Successfully put {self.object_name} in {self.target}.",
            result=True,
            error_flag=status.error_flag,
        )

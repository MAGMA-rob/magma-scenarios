# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict

from magma_core.simulation.data_structures import RobotToolStatus, ToolErrorFlag
from magma_core.simulation.skills import BaseSkill, CallTick, MessTick, SkillSpec, SkillTick


class CycleSkill(BaseSkill):
    spec = SkillSpec(
        name="fullfill_package",
        description=(
            "Put every requested product into the held package."
        ),
        required_tools=["take", "put"],
        argument_schema={
            "objects": {
                "type": "list",
                "description": "The ordered list of product types to put in the package.",
            }
        },
    )

    def __init__(self, arguments: Dict) -> None:
        super().__init__(arguments)
        self.objects = arguments.get("objects", [])
        self.current_object_index = 0
        self.failure_count = 0
        self.current_tool = "take"

    def start(self) -> SkillTick:
        if not isinstance(self.objects, list):
            return MessTick(
                message="objects must be a list of product types.",
                result=False,
                error_flag=ToolErrorFlag.BAD_CALL,
            )
        return self._call_current_tool()

    def tick(self, status: RobotToolStatus) -> SkillTick:
        if not status.result:
            self.failure_count += 1
            if self.failure_count >= 2:
                return MessTick(
                    message=(
                        "Delivery cycle aborted after two failures while processing "
                        f"{self.objects[self.current_object_index]}: {status.mess}"
                    ),
                    result=False,
                    error_flag=status.error_flag,
                )
            return self._call_current_tool()

        self.failure_count = 0

        if self.current_tool == "take":
            self.current_tool = "put"
            return self._call_current_tool()

        if self.current_tool == "put":
            self.current_object_index += 1
            if self.current_object_index < len(self.objects):
                self.current_tool = "take"
                return self._call_current_tool()

        return MessTick(
            message=(
                f"Successfully put {self.objects} in the held package."
            ),
            result=True,
            error_flag=status.error_flag,
        )

    def _call_current_tool(self) -> CallTick:
        if self.current_tool == "take":
            return CallTick(
                tool_name="take",
                arguments={"type_obj": self.objects[self.current_object_index]},
            )
        if self.current_tool == "put":
            return CallTick(tool_name="put", arguments={"target": "package"})
        raise RuntimeError(f"Unknown delivery skill step {self.current_tool!r}")
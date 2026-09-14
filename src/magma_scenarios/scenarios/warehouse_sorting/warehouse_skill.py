# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict

from magma_core.simulation.data_structures import RobotToolStatus, ToolErrorFlag
from magma_core.simulation.skills import BaseSkill, CallTick, MessTick, SkillSpec, SkillTick


class CycleSkill(BaseSkill):
    spec = SkillSpec(
        name="launch_cycle",
        description="Sort every requested object into its assigned target area.",
        required_tools=["take_obj", "depose"],
        argument_schema={
            "assignment": {
                "type": "dict",
                "description": "A mapping from each object name to its target area.",
            },
        },
    )

    def __init__(self, arguments: Dict) -> None:
        super().__init__(arguments)
        self.assignment = arguments.get("assignment", {})
        self.objects = list(self.assignment) if isinstance(self.assignment, dict) else []
        self.current_object_index = 0
        self.failure_count = 0
        self.current_tool = "take_obj"

    def start(self) -> SkillTick:
        if not isinstance(self.assignment, dict) or not self.assignment:
            return MessTick(
                message="The cycle assignment must be a non-empty dictionary.",
                result=False,
                error_flag=ToolErrorFlag.BAD_CALL,
            )
        if not all(
            isinstance(obj_name, str) and isinstance(target_name, str)
            for obj_name, target_name in self.assignment.items()
        ):
            return MessTick(
                message="Every object and target area in the assignment must be a string.",
                result=False,
                error_flag=ToolErrorFlag.BAD_CALL,
            )
        return self._take_current_object()

    def tick(self, status: RobotToolStatus) -> SkillTick:
        if not status.result and "already sorted" in status.mess:
            self.current_tool="depose"
        elif not status.result:
            self.failure_count += 1
            if self.failure_count >= 2:
                return MessTick(
                    message=(
                        "Sorting cycle aborted after two failures while processing "
                        f"{self.objects[self.current_object_index]}: {status.mess}"
                    ),
                    result=False,
                    error_flag=status.error_flag,
                )

            if self.current_tool == "take_obj":
                return self._take_current_object()

            if self.current_tool == "depose":
                return self._depose_current_object()

            raise RuntimeError(f"Unknown warehouse sorting skill step {self.current_tool!r}")

        self.failure_count = 0

        if self.current_tool == "take_obj":
            self.current_tool = "depose"
            return self._depose_current_object()

        self.current_object_index += 1
        if self.current_object_index < len(self.objects):
            self.current_tool = "take_obj"
            return self._take_current_object()

        return MessTick(
            message=f"Successfully sorted {len(self.assignment)} objects.",
            result=True,
            error_flag=status.error_flag,
        )

    def _take_current_object(self) -> CallTick:
        return CallTick(
            tool_name="take_obj",
            arguments={"obj": self.objects[self.current_object_index]},
        )

    def _depose_current_object(self) -> CallTick:
        current_object = self.objects[self.current_object_index]
        return CallTick(
            tool_name="depose",
            arguments={"target": self.assignment[current_object]},
        )

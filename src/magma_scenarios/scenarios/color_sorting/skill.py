# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict

from magma_core.simulation.data_structures import RobotToolStatus, ToolErrorFlag
from magma_core.simulation.skills import BaseSkill, CallTick, MessTick, SkillSpec, SkillTick


class CycleSkill(BaseSkill):
    spec = SkillSpec(
        name="launch_cycle",
        description=(
            " "
        ),
        required_tools=["take", "put", "mark_delivery"],
        argument_schema={
            "recipe": {
                "type": "list",
                "description": "The ordered list of product types to deliver.",
            },
            "delivery_number": {
                "type": "int",
                "description": "The number of deliveries to record.",
            },
            "manufacturing_order": {
                "type": "str",
                "description": "The manufacturing order of the delivery cycle.",
            },
        },
    )

    def __init__(self, arguments: Dict) -> None:
        super().__init__(arguments)
        self.recipe = arguments.get("recipe", [])
        self.delivery_number = arguments.get("delivery_number", -1)
        self.manufacturing_order = arguments.get("manufacturing_order", "")
        self.current_product_index = 0
        self.failure_count = 0
        self.current_tool = "take"

    def start(self) -> SkillTick:
        if not isinstance(self.recipe, list) or len(self.recipe) == 0:
            return MessTick(
                message="The delivery recipe must contain at least one product.",
                result=False,
                error_flag=ToolErrorFlag.BAD_CALL,
            )
        if not all(isinstance(product_type, str) for product_type in self.recipe):
            return MessTick(
                message="Every product type in the delivery recipe must be a string.",
                result=False,
                error_flag=ToolErrorFlag.BAD_CALL,
            )
        if not isinstance(self.delivery_number, int) or self.delivery_number == 0:
            return MessTick(
                message="delivery_number must be a non-zero integer.",
                result=False,
                error_flag=ToolErrorFlag.BAD_CALL,
            )
        if not isinstance(self.manufacturing_order, str) or not self.manufacturing_order:
            return MessTick(
                message="manufacturing_order must be a non-empty string.",
                result=False,
                error_flag=ToolErrorFlag.BAD_CALL,
            )
        return self._take_current_product()

    def tick(self, status: RobotToolStatus) -> SkillTick:
        if not status.result:
            self.failure_count += 1
            if self.failure_count >= 2:
                failed_step = (
                    "mark_delivery"
                    if self.current_tool == "mark_delivery"
                    else self.recipe[self.current_product_index]
                )
                return MessTick(
                    message=(
                        f"Delivery cycle aborted after two failures while processing "
                        f"{failed_step}: {status.mess}"
                    ),
                    result=False,
                    error_flag=status.error_flag,
                )

            if self.current_tool == "take":
                return self._take_current_product()

            if self.current_tool == "put":
                return CallTick(tool_name="put", arguments={})

            if self.current_tool == "mark_delivery":
                return CallTick(
                    tool_name="mark_delivery",
                    arguments={
                        "delivery_number": self.delivery_number,
                        "manufacturing_order": self.manufacturing_order,
                    },
                )

            raise RuntimeError(f"Unknown delivery skill step {self.current_tool!r}")

        self.failure_count = 0

        if self.current_tool == "take":
            self.current_tool = "put"
            return CallTick(tool_name="put", arguments={})

        if self.current_tool == "put":
            self.current_product_index += 1
            if self.current_product_index < len(self.recipe):
                self.current_tool = "take"
                return self._take_current_product()

            self.current_tool = "mark_delivery"
            return CallTick(
                tool_name="mark_delivery",
                arguments={
                    "delivery_number": self.delivery_number,
                    "manufacturing_order": self.manufacturing_order,
                },
            )

        return MessTick(
            message=(
                f"Successfully completed {self.delivery_number} deliveries under "
                f"manufacturing order {self.manufacturing_order}."
            ),
            result=True,
            error_flag=status.error_flag,
        )

    def _take_current_product(self) -> CallTick:
        return CallTick(
            tool_name="take",
            arguments={"type_obj": self.recipe[self.current_product_index]},
        )

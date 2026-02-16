# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tools import register_tool
from magma_core.base.data_structures import ToolExecution

from .base import HallTool

from typing import Dict

class SimplifiedHallTool(HallTool):

    @register_tool( 
        description="Return the list of detected object in each hall",
        params_spec={}
    )
    def detect_object(self, obs: Dict, env_id: int, params: Dict) -> ToolExecution:
        return super().detect_object(obs, env_id, params)

    @register_tool(
        description="Take specified objects from the specified hall. If objects not present, return errors",
        params_spec={
            "objects" : {"description":"The list of objects to take. They must be in the hall.","type":list},
            "hall" : {"description":"The name of the hall where the robot must pick objects","type":str}
        }
    )
    def take_objects_from_hall(self, obs: Dict, env_id: int, params: Dict) -> ToolExecution:
        ...

    @register_tool(
        description="Depose the specified objects from the robot bag to the specified hall. Objects must be in the robot bag.",
        params_spec={
            "objects" : {"description":"The list of objects to depose. They must be in the robot bag.","type":list},
            "hall" : {"description":"The name of the hall where the robot must depose objects","type":str}
        }
    )
    def depose_objects_to_hall(self, obs: Dict, env_id: int, params: Dict) -> ToolExecution:
        ...

    
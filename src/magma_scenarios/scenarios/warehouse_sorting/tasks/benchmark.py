# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from magma_core.base.tasks import BaseBenchmarkTask

from pathlib import Path

from ..tools import WithoutManufacturingOrder, WithManufacturingOrder
from .att import OBJECTS, AREAS

class WS_Simp_Benchmark(BaseBenchmarkTask):
    """
    Class to create a Sorting task in a factory. Robot knows some objects and area. Users gives constraint about objects assignment.
    The robot must solves these constraint to complete multiple cycle.
    """
    name : str = "Benchmark Warehouse sorting simplified [PAPER VERSION]"
    env_id : str = "SortingCubesWarehouse-v1"

    randomized_config_path = str(Path(__file__).joinpath("config.yaml"))

    Tools_cls = WithoutManufacturingOrder

    all_task_attributes = {
        "objects" : OBJECTS[:3],
        "target_areas" : AREAS[:3]
    }


class WS_Benchmark(BaseBenchmarkTask):
    """
    Class to create a Sorting task in a factory. Robot knows some objects and area. Users gives constraint about objects assignment.
    The robot must solves these constraint to complete multiple cycle.
    """
    name : str = "Benchmark Warehouse sorting"
    env_id : str = "SortingCubesWarehouse-v1"

    randomized_config_path = str(Path(__file__).joinpath("config.yaml"))

    Tools_cls = WithManufacturingOrder

    all_task_attributes = {
        "objects" : OBJECTS,
        "target_areas" : AREAS
    }
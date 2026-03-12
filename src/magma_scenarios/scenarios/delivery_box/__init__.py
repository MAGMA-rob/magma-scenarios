# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

SCENARIO_NAME = "delivery_box"

TASK_DEFINITIONS = {}

TASK_PRESETS = {
    "BenchDeliveryTask": "tasks.simple_delivery:BenchDeliveryTask",
    "EvolvingDelivery": "tasks.simple_delivery:EvolvingDelivery",
}

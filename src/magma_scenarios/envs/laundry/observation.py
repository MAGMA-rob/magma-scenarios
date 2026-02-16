# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import typing

import torch


class ObjectObservation(typing.TypedDict):
    pose: torch.Tensor

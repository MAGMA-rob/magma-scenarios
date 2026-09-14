# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Dict, Union

import numpy as np
import sapien, random
import torch

from mani_skill.utils.structs import Pose
from mani_skill.utils.registration import register_env

from .six_cubes_two_boxes_on_table import SixCubesTwoBoxesOnTable


@register_env("PartialSixCubesTwoBoxesOnTable", max_episode_steps=200)
class PartialSixCubesTwoBoxesOnTable(SixCubesTwoBoxesOnTable):
    """
    Task Description
    ----------------
    The task is to place 3 green and 3 yellow cubes initially lying on a table in boxes
    of the same color. Here some cubes are randomly already sorted.

    Randomizations
    --------------
    Positions of cube are randomized.
    """

    size_box = 0.3

        
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        return super()._initialize_episode(env_idx, options)

    def _get_obs_extra(self, info: Dict):
        return super()._get_obs_extra(info)

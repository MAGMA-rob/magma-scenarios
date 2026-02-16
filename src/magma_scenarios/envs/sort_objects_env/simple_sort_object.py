# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from .sort_object_base import SourceSortObjectEnv

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.structs import Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.building.actors.ycb import get_ycb_builder


# register the environment by a unique ID and specify a max time limit. Now once this file is imported you can do gym.make("CustomEnv-v0")
@register_env("SortObject-v1", max_episode_steps=200)
class SortObjectEnvSimple(SourceSortObjectEnv):
    """
    Task Description
    ----------------
    No obstacle
    """

    def _load_scene(self, options: dict):
         # we use a prebuilt scene builder class that automatically loads in a floor and table.
        super()._load_scene(options=options)

        self.build_an_ycb("013_apple","apple",sapien.Pose(p=[0.04256231, 0.1, 0.015283]))
        self.build_an_ycb("024_bowl","bowl")
        self.build_an_ycb("030_fork","fork",sapien.Pose(p=[0.1, 0.1, 0.1]))
        self.build_an_ycb("011_banana","banana",sapien.Pose(p=[-0.1, 0.0050657, 0.015283]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx=env_idx,options=options)
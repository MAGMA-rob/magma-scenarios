# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

import sys
import types
from pathlib import Path


SCENARIOS_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SCENARIOS_ROOT.parent

for src in [
    SCENARIOS_ROOT / "src",
    WORKSPACE_ROOT / "magma-core-dev" / "src",
]:
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _ensure_sapien_stub() -> None:
    try:
        import sapien  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        sapien = types.ModuleType("sapien")

        class Pose:
            def __init__(self, *args, **kwargs):
                self.p = kwargs.get("p", [0, 0, 0])
                self.q = kwargs.get("q", [1, 0, 0, 0])

        sapien.Pose = Pose
        sys.modules["sapien"] = sapien


def _ensure_mani_skill_stub() -> None:
    try:
        import mani_skill.agents.base_agent  # type: ignore  # noqa: F401
        import mani_skill.utils.building  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        mani_skill = types.ModuleType("mani_skill")
        agents = types.ModuleType("mani_skill.agents")
        base_agent = types.ModuleType("mani_skill.agents.base_agent")
        utils = types.ModuleType("mani_skill.utils")
        building = types.ModuleType("mani_skill.utils.building")
        common = types.ModuleType("mani_skill.utils.common")
        envs = types.ModuleType("mani_skill.envs")
        scene = types.ModuleType("mani_skill.envs.scene")

        class BaseAgent:
            pass

        class ArticulationBuilder:
            pass

        class URDFLoader:
            pass

        class ManiSkillScene:
            pass

        def to_numpy(value):
            return value

        base_agent.BaseAgent = BaseAgent
        building.ArticulationBuilder = ArticulationBuilder
        building.URDFLoader = URDFLoader
        common.to_numpy = to_numpy
        scene.ManiSkillScene = ManiSkillScene
        agents.base_agent = base_agent
        utils.building = building
        utils.common = common
        envs.scene = scene
        mani_skill.agents = agents
        mani_skill.utils = utils
        mani_skill.envs = envs

        sys.modules["mani_skill"] = mani_skill
        sys.modules["mani_skill.agents"] = agents
        sys.modules["mani_skill.agents.base_agent"] = base_agent
        sys.modules["mani_skill.utils"] = utils
        sys.modules["mani_skill.utils.building"] = building
        sys.modules["mani_skill.utils.common"] = common
        sys.modules["mani_skill.envs"] = envs
        sys.modules["mani_skill.envs.scene"] = scene


def _ensure_transforms3d_stub() -> None:
    try:
        import transforms3d.euler  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        transforms3d = types.ModuleType("transforms3d")
        euler = types.ModuleType("transforms3d.euler")
        euler.euler2quat = lambda *args, **kwargs: [1, 0, 0, 0]
        transforms3d.euler = euler
        sys.modules["transforms3d"] = transforms3d
        sys.modules["transforms3d.euler"] = euler


def _ensure_torch_stub() -> None:
    try:
        import torch  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        torch = types.ModuleType("torch")
        torch_tensor = types.ModuleType("torch._tensor")

        class Tensor:
            def __init__(self, *args, **kwargs):
                self.shape = (0,)
                self.device = "cpu"

            def item(self):
                return 0

        def _tensor(*args, **kwargs):
            return Tensor()

        torch.Tensor = Tensor
        torch_tensor.Tensor = Tensor
        torch.__codex_stub__ = True
        torch_tensor.__codex_stub__ = True
        torch.tensor = _tensor
        torch.ones = _tensor
        torch.minimum = lambda a, b: a
        torch.zeros_like = _tensor
        torch.int32 = "int32"

        sys.modules["torch"] = torch
        sys.modules["torch._tensor"] = torch_tensor


_ensure_sapien_stub()
_ensure_mani_skill_stub()
_ensure_transforms3d_stub()
_ensure_torch_stub()

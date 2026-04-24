from pathlib import Path
import sapien

from magma_core.base.tasks import BaseBenchmarkTask
from .simple_tool import MakingCoffeeTool
from .attributes import att, people, teams, loaded_capsule_pose, dropped_mug_pose
import sapien

class CoffeeBenchmark(BaseBenchmarkTask):
    """
    Benchmark for the coffee making scenario.
    Tests constraint following, multi-step reasoning, and team-based preference rules.
    """

    name: str = "Benchmark Coffee Making"
    env_id: str = "MakeCoffee-v1"  

    Tools_cls = MakingCoffeeTool 

    env_options = {}

    tools_constant = {
        "teams": {
            team: people[i * (len(people) // len(teams)): (i + 1) * (len(people) // len(teams))]
            for i, team in enumerate(teams)
        },
        "loaded_capsule_pose": loaded_capsule_pose,
        "dropped_mug_pose" : dropped_mug_pose,
        "base_pose" : sapien.Pose(p = [-0.1,0,0.4],q = [0,1,0,0])
    }

    all_task_attributes = att


"""
Avec len(people) = 23 et len(teams) = 5, le pas est 23 // 5 = 4 :

DISCO [0:4] : Smith, Anderson, Clark, Wright
GEPETTO [4:8] : Mitchell, Johnson, Thomas, Rodriguez
RAP [8:12] : Lopez, Perez, Williams, Jackson
RIS [12:16] : Lewis, Hill, Roberts, Jones
MAC [16:20] : White, Lee, Scott, Turner

Et Brown, Harris, Walker ([20:23]) ne sont dans aucune équipe.
"""
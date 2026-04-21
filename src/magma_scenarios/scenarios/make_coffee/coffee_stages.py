from magma_core.base.stage import BaseTaskStage, ConstraintBaseStage, BaseStageComposite, AskingBaseStage
from magma_core.utils.env_utils import is_object_inside_target
from magma_core.base.data_structures import StageState, UserInstruction, EmptyInstruction, Log, Situation
from magma_core.base.goals import BaseGoal

from magma_scenarios.utils import sapien_to_tensor

from .attributes import att, dropped_mug_pose, loaded_capsule_pose

import sapien, torch, random
from collections import defaultdict
from typing import List, Dict, Any

class MugAndCapsuleGoal(BaseGoal):
    """Specific coffee scenario goal. Adding a relative add to the position for the target"""

    def __init__(self, capsule : str) -> None:
        super().__init__("Mug And Capsule positioned",None)
        self.capsule_target = sapien_to_tensor(loaded_capsule_pose)
        self.mug_target = sapien_to_tensor(dropped_mug_pose)
        self.capsule = capsule

    def verify(self, obs: Dict) -> torch.Tensor:
        device = obs["extra"]["coffee_maker"].device

        # check if the waited capsule is placed
        is_capsule_loaded = is_object_inside_target(
            obs["extra"][self.capsule], 
            torch.add(obs["extra"]["coffee_maker"][:,:7], self.capsule_target.to(device)),
            0.03)
        
        # check if the mug is placed
        is_mug_dropped = is_object_inside_target(
            obs["extra"]["mug"], 
            torch.add(obs["extra"]["coffee_maker"][:,:7], self.mug_target.to(device)),
            0.03)
        
        return (is_capsule_loaded & is_mug_dropped).int()

class ConstraintCoffeeStage(ConstraintBaseStage):

    def __init__(self, constraint: str) -> None:
        mem = [
            "You are in charge of making a coffee.",
            "To prepare a coffee you must have placed the mug and loaded the capsule before pressing the start button."
            ]
        super().__init__(constraint, mem, att)

class RefuseCoffee(BaseTaskStage):

    acceptance_steps = 0
    target_steps = 1

    def __init__(self, instruction, verif_prompt : str) -> None:
        super().__init__([], False, "")

        self.situation = Situation(
            memory=[],
            preserved_memory_indices=[],
            attributes=att,
            flag_answer_to_user=False,
            instruction=UserInstruction(instruction),
        )

        self.verification_prompt = verif_prompt

class MakeOneCoffeStage(BaseTaskStage):

    target_steps = 3
    acceptance_steps = 1

    def __init__(self, capsule : str, instruction : str, add_memory : List[str], flag_answer : bool) -> None:
        if not capsule in att["coffee_pod"]:
            raise ValueError(f"capsule {capsule} is not a known taste. Available are : {att['coffee_pod']}")

        super().__init__(
            [MugAndCapsuleGoal(capsule)],
            True, 
            f"The goal of this stage is to add the mug to the machine, load a {capsule} pod and start the coffee machine."
        )

        mem = [
            "You are in charge of making a coffee.",
            "To prepare a coffee you must have placed the mug and loaded the capsule before pressing the start button."
            ]
        mem.extend(add_memory)

        self.situation = Situation(
            memory=mem,
            instruction=UserInstruction(instruction) if instruction != "none" else EmptyInstruction(),
            attributes=att,
            flag_answer_to_user=flag_answer,
            preserved_memory_indices=[]
        )    
    
    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        """
        Check if the press button correctly happens after mug and pods placed.
        """        
        is_pods_load = False
        is_mug_placed = False

        for l in stage_log:
            task_name = l.function
            if task_name == "load_capsule": is_pods_load = True
            if task_name == "place_mug": is_mug_placed = True
            if task_name == "press_button":
                if is_mug_placed and is_pods_load:
                    return 1
                return -1
        return 0
    
    def combine_stage_completion(self, task_completion: int, log_completion: int) -> int:
        """
        Here it's a special case, if the log return 1 and the env 0, it's a -1 (because maybe the pod or the mug was not correctly placed - the action failed and we press)
        """
        if log_completion == 1:
            if task_completion == 1:
                return 1
            return -1 # case where the env failed
        if task_completion == -1 or log_completion == -1:
            return -1
        return 0
    
class CoffeeCompositeStage(BaseStageComposite):

    def __init__(self, coffee_number : Dict[str,int]) -> None:
        instruction = UserInstruction(self._build_main_instruction(coffee_number))
        stages = []
        number_per_stage = []
        for pod, val in coffee_number.items():
            stages.append(MakeOneCoffeStage(pod, "none", [],False))
            number_per_stage.append(val)
        situation = Situation(
            [],
            [],
            att,
            instruction,
            flag_answer_to_user=True
        )
        super().__init__(stages, number_per_stage, situation)
        self.coffee_desired = coffee_number

    def _build_main_instruction(self, repartition : Dict[str,int])->str:        
        ins = "Hello, make me "
        for i, (pod, val) in enumerate(repartition.items()):
            ins += f"{val} {pod} coffee"
            if i < len(repartition)-1:
                ins += " and "
        return ins + "."
    
    def _count_completion(self, full_log: List[Log]) -> Dict[str,int]:
        cpt = {k:0 for k in self.coffee_desired}
        pod = None
        for l in full_log:
            if l.function == "load_capsule":
                pod = l.content
            if l.function == "press_button":
                if pod is None:
                    raise RuntimeError("Impossible fail")
                if not pod in cpt:
                    raise RuntimeError(f"A no desired pod have been done : {pod} --> {cpt}")
                cpt[pod] += 1
                pod = None
        return cpt

    def is_fully_completed(self, full_log: List[Log]) -> int:
        cpt = self._count_completion(full_log)
        
        for pod, val in cpt.items():
            if val < self.coffee_desired[pod]:
                return 0
            elif val > self.coffee_desired[pod]:
                return -1
        
        return 1
    
    def get_stage_state(
            self,
            agent_step: int,
            log: List[Log],
            recovery_extra_steps: int = 0,
        ) -> StageState:
        n = 0
        completion = 0

        if len(log) > 0:
            stage_id = log[-1].stage_id
            for l in log:
                if l == stage_id:
                    break
                n+=1
            cpt = self._count_completion(log[n:])
            completion = sum([v for k,v in cpt.items()])

        total = sum([v for k,v in self.coffee_desired.items()])

        if completion == total:
            if agent_step <= completion * 3 + 1 + recovery_extra_steps:
                return StageState.OPTIMAL

        max_acceptable = completion * 4 + recovery_extra_steps
        if agent_step >= max_acceptable: #hardcoded value
            return StageState.EXCEEDED

        return StageState.ACCEPTABLE

class AskTeamStage(AskingBaseStage):

    acceptance_steps = 0
    target_steps = 2

    def __init__(self, requested_team : str, member_list : List[str]) -> None: 
        super().__init__(
            question=f"who is in team {requested_team} ?",
            answer=f"{member_list} are in team {requested_team}",
            memory=[],
            attributes=att,
            linked_to_prev=True,
            allow_tools_before_answer=True
        )

## ask team stage version person stage
class  AskPersonStage(AskingBaseStage):

    acceptance_steps = 0
    target_steps = 0

    def __init__(self,requested_person : str, team_associated : str) -> None:
        super().__init__(
            question = f"what team {requested_person} belongs to ?",
            answer=f"{requested_person} belongs to {team_associated}",
            memory=[],
            attributes=att,
            linked_to_prev=True,
            allow_tools_before_answer=True
        )
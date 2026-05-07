from magma_core.base.stage import BaseTaskStage, ConstraintBaseStage, BaseStageComposite, AskingBaseStage
from magma_core.utils.env_utils import is_object_inside_target
from magma_core.base.data_structures import StageState, UserInstruction, EmptyInstruction, Log, Situation
from magma_core.base.goals import BaseGoal

from magma_scenarios.utils import sapien_to_tensor

from .attributes import att, dropped_mug_pose, loaded_capsule_pose
from .coffee_errors import GraspCapsuleFailureError

import sapien, torch, random
from collections import defaultdict
from typing import List, Dict, Any, Optional

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
        self.capsule = capsule
    
    def verif_log_completion(self, stage_log : List[Log], full_log : List[Log]) -> int:
        """
        Check if the press button correctly happens after mug and pods placed.
        """
        loaded_pods = []
        is_mug_placed = False

        for l in stage_log:
            task_name = l.function
            if task_name == "load_capsule": loaded_pods.append(l.content)
            if task_name == "place_mug": is_mug_placed = True
            if task_name == "press_button":
                if is_mug_placed and loaded_pods == [self.capsule]:
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
        loaded_pods = []
        for l in full_log:
            if l.function == "load_capsule":
                loaded_pods.append(l.content)
            if l.function == "press_button":
                if len(loaded_pods) == 0:
                    raise RuntimeError("Impossible fail")
                if len(loaded_pods) > 1:
                    raise RuntimeError(f"Too many pods have been loaded before pressing: {loaded_pods}")
                pod = loaded_pods[0]
                if not pod in cpt:
                    raise RuntimeError(f"A no desired pod have been done : {pod} --> {cpt}")
                cpt[pod] += 1
                loaded_pods = []
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

def _join_sentence_parts(values: List[str]) -> str:
    if len(values) == 0:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


class AskPeopleInTeamStage(AskingBaseStage):

    acceptance_steps = 0
    target_steps = 2

    def __init__(self, requested_team : str, member_list : List[str]) -> None: 
        if len(member_list) == 0:
            raise ValueError("member_list must not be empty")
        members = _join_sentence_parts(member_list)
        be_verb = "is" if len(member_list) == 1 else "are"
        super().__init__(
            question=f"Who is in team {requested_team}?",
            answer=f"{members} {be_verb} in team {requested_team}.",
            memory=[],
            attributes=att,
            linked_to_prev=True,
            allow_tools_before_answer=True,
        )
        self.stage_goal_description = f"The goal of the stage is to ensure that the model call the tool to fetch team information before answering that : {self.verification_prompt}"


class AskTeamsForPeopleStage(AskingBaseStage):

    acceptance_steps = 0

    def __init__(self, requested_people : List[str], teams_associated : List[str]) -> None:
        if len(requested_people) != len(teams_associated):
            raise ValueError(
                "requested_people and teams_associated must have the same length"
            )
        if len(requested_people) == 0:
            raise ValueError("requested_people must not be empty")

        if len(requested_people) == 1:
            question = f"What team does {requested_people[0]} belong to?"
        else:
            question = f"In which teams are {_join_sentence_parts(requested_people)}?"

        answer_parts = [
            f"{person} is in team {team}"
            for person, team in zip(requested_people, teams_associated)
        ]
        super().__init__(
            question=question,
            answer=_join_sentence_parts(answer_parts) + ".",
            memory=[],
            attributes=att,
            linked_to_prev=True,
            allow_tools_before_answer=True
        )
        self.target_steps = len(requested_people) + 1
        self.stage_goal_description = f"The goal of the stage is to ensure that the model call the tool to fetch team information before answering that : {self.verification_prompt}"


class AskCoffeePreferenceForUserStage(AskingBaseStage):

    acceptance_steps = 0
    target_steps = 1

    def __init__(self, requested_person: str, coffee_preference: Optional[str]) -> None:
        if requested_person == "":
            raise ValueError("requested_person must not be empty")

        question = random.choice([
            f"What coffee does {requested_person} like?",
            f"What is {requested_person}'s coffee preference?",
            f"Which coffee should I prepare for {requested_person}?",
        ])

        if coffee_preference is None:
            answer = f"{requested_person} does not have any known coffee preference."
        else:
            answer = f"{requested_person} likes {coffee_preference} coffee."

        super().__init__(
            question=question,
            answer=answer,
            memory=[],
            attributes=att,
            linked_to_prev=True,
        )


class AskTeamCoffeePreferencesStage(AskingBaseStage):

    acceptance_steps = 0
    target_steps = 2

    def __init__(
            self,
            requested_team: str,
            team_members: List[str],
            known_preferences: Dict[str, str],
            focus: str = "all"
        ) -> None:
        if len(team_members) == 0:
            raise ValueError("team_members must not be empty")

        question = self._build_question(requested_team, focus)
        answer = self._build_answer(requested_team, team_members, known_preferences, focus)

        super().__init__(
            question=question,
            answer=answer,
            memory=[],
            attributes=att,
            linked_to_prev=True,
            allow_tools_before_answer=True,
            allowed_tools=["people_from_team"],
        )
        self.stage_goal_description = f"The goal of the stage is to ensure that the model call the tool to fetch team information before answering that : {self.verification_prompt}"

    def _build_question(self, requested_team: str, focus: str) -> str:
        if focus == "all":
            return f"What are the coffee preferences of the members of team {requested_team}?"
        if focus == "unknown":
            return f"Which members of team {requested_team} do not have a known coffee preference?"
        return f"Which members of team {requested_team} like {focus} coffee?"

    def _build_answer(
            self,
            requested_team: str,
            team_members: List[str],
            known_preferences: Dict[str, str],
            focus: str,
        ) -> str:
        if focus == "all":
            return self._build_full_preference_answer(requested_team, team_members, known_preferences)
        if focus == "unknown":
            return self._build_unknown_preference_answer(requested_team, team_members, known_preferences)
        return self._build_filtered_preference_answer(requested_team, team_members, known_preferences, focus)

    def _build_full_preference_answer(
            self,
            requested_team: str,
            team_members: List[str],
            known_preferences: Dict[str, str],
        ) -> str:
        parts = [
            f"{name} likes {known_preferences[name]} coffee"
            for name in team_members
            if name in known_preferences
        ]
        unknown_people = [name for name in team_members if name not in known_preferences]

        if len(unknown_people) == len(team_members):
            return f"No member of team {requested_team} has a known coffee preference."

        if len(unknown_people) == 1:
            parts.append(f"{unknown_people[0]} does not have any known coffee preference")
        elif len(unknown_people) > 1:
            parts.append(
                f"{_join_sentence_parts(unknown_people)} do not have any known coffee preference"
            )

        return f"In team {requested_team}, {_join_sentence_parts(parts)}."

    def _build_unknown_preference_answer(
            self,
            requested_team: str,
            team_members: List[str],
            known_preferences: Dict[str, str],
        ) -> str:
        unknown_people = [name for name in team_members if name not in known_preferences]
        if len(unknown_people) == 0:
            return f"All members of team {requested_team} have a known coffee preference."
        if len(unknown_people) == 1:
            return f"{unknown_people[0]} does not have any known coffee preference in team {requested_team}."
        return (
            f"{_join_sentence_parts(unknown_people)} do not have any known coffee preference "
            f"in team {requested_team}."
        )

    def _build_filtered_preference_answer(
            self,
            requested_team: str,
            team_members: List[str],
            known_preferences: Dict[str, str],
            focus: str,
        ) -> str:
        matching_people = [
            name
            for name in team_members
            if known_preferences.get(name) == focus
        ]
        if len(matching_people) == 0:
            return f"No member of team {requested_team} has a known preference for {focus} coffee."
        if len(matching_people) == 1:
            return f"{matching_people[0]} likes {focus} coffee in team {requested_team}."
        return f"{_join_sentence_parts(matching_people)} like {focus} coffee in team {requested_team}."

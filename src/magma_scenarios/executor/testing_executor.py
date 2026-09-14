# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict, List, Union, Tuple, Optional
from collections import OrderedDict
from magma_core.simulation.envs import DefaultEnv
from magma_core.simulation.tasks.base_task import BaseTask
import torch
import copy

from magma_core.simulation.executor import SingleTaskExecutor
from magma_core.simulation.data_structures import EnvToolContext, Log, ToolStatus
from magma_core.domain import Call
from magma_core.utils.global_utils import (
    apply_env_state_updates,
    extract_env_state_val,
    batch_set_value,
)

class ToolsTestingExecutor(SingleTaskExecutor):
    """
    Tools executor is the main class responsible for transforming LLM calls into actions using specific defined class.
    """

    # Only for Evaluation Mode
    _eval_envs : Dict[int,EnvToolContext]

    attributes : Dict

    def __init__(
            self,
            planner_endpoint : str,
            nb_env : int = 1,
            randomized : bool = False,
        ):
        
        super().__init__(
            nb_env,
            planner_endpoint=planner_endpoint,
            ollama_worker=None,
            nb_randomization=int(randomized),
            gui=True,
        )

        self._eval_envs : Dict[int,EnvToolContext] = {}
        self.attributes = {}

    def _find_next_non_text_stage(
            self,
            start_stage_id: int,
            display_skipped_instruction: bool = True,
            display_next_stage_after_skip: bool = False,
        ) -> Optional[int]:
        """
        Return the next non text-only stage id starting from ``start_stage_id``.
        If all remaining stages are text-only, return None.
        """
        nb_stages = self.task_ref.get_nb_total_stage()
        stage_id = start_stage_id
        skipped_stage = False

        while stage_id < nb_stages:
            if not self.task_ref.is_stage_text_only(stage_id):
                if skipped_stage and display_next_stage_after_skip:
                    instruction = self.get_instruction(stage_id)
                    print(f"[EXECUTOR] Next Stage {stage_id} with instruction {instruction.get_content()}")
                return stage_id
            if display_skipped_instruction:
                instruction = self.get_instruction(stage_id)
                print(f"[EXECUTOR] Skip Stage {stage_id} with instruction {instruction.get_content()}")
            skipped_stage = True
            stage_id += 1

        return None
   
    ################ public function

    def check_env_state(self, obs : Dict):
        """
        Used for Checking if env has passed the stage in eval mode.

        Return a list of 0 (running), 1 (finish), -1 (error) representing the status of all envs.
        """
        out = []
        stage_id = self._eval_envs[0].current_task_stage
        last_stage_id = self.task_ref.get_nb_total_stage() - 1

        if self.task_ref.is_stage_text_only(stage_id):
            if stage_id != last_stage_id:
                raise RuntimeError(f"Unexpected text-only current stage {stage_id} in testing executor")
            print("FINISHED TASK")
            return [1] * self.nb_env

        env_ids = list(range(self.nb_env))
        env_verif = self.task_ref.verif_stage_env_completion(stage_id, obs, env_ids=env_ids).cpu().tolist()
        st = self.env.unwrapped.get_state_dict().copy()
        env_ids_tensor = torch.tensor([env_ids])
        for i in range(self.nb_env):
            full_log, stage_log = self._get_logs(i)
            log_verif = self.task_ref.verif_stage_log_completion(
                stage_id=stage_id,
                full_log=full_log,
                stage_log=stage_log,
                composite_progress=self._eval_envs[i].composite_progress,
            )
            out.append(self.task_ref.combine_stage_verif_scores(stage_id, env_verif[i], log_verif))
            if self.task_ref.should_reset_same_stage(stage_id):
                batch_set_value(
                    state_env=st,
                    env_ids=env_ids_tensor,
                    template=self.task_ref._default_env_state
                )
                self._eval_envs[i].stage_log_start_idx = len(self._eval_envs[i].logs)

        if all([score == 1 for score in out]):
            if stage_id != last_stage_id:
                next_stage_id = self._find_next_non_text_stage(stage_id + 1)
                self._pass_to_the_next_stage(
                    self.task_ref,
                    stage_id,
                    env_ids,
                    st,
                )

                if next_stage_id is None:
                    for i in range(self.nb_env):
                        self._eval_envs[i].current_task_stage = last_stage_id
                        self._eval_envs[i].stage_log_start_idx = len(self._eval_envs[i].logs)
                        self._eval_envs[i].error_state = {}
                    print("FINISHED TASK")
                else:
                    instruction = self.get_instruction(next_stage_id)
                    print(f"NEXT STAGE : {next_stage_id} with instruction {instruction.get_content()}")

                    for i in range(self.nb_env):
                        # Specific modifications
                        self._eval_envs[i].current_task_stage = next_stage_id
                        self._eval_envs[i].stage_log_start_idx = len(self._eval_envs[i].logs)
                        # A new stage must rebuild its own active error profile.
                        self._eval_envs[i].error_state = {}
            else:
                print("FINISHED TASK")

        self.env.unwrapped.set_state_dict(st)

        return out
    
    def initialize(
            self,
            task_ref: BaseTask,
            build_first_stage: bool = True,
            obs_mode: str = "state_dict",
            sim_backend: str = "auto"
        ) -> DefaultEnv:
        out = super().initialize(
            task_ref, build_first_stage, obs_mode, sim_backend
        )
        self.attributes = self.task_ref.get_init_attributes()
        return out
    
    def compute_actions(self, tools_call : Dict[int,List[Call]]):
        """
        Take a batch of tools_calls. It's a dict where each key is a node_id associate with a dict.
        In case of GENERATION mode, the value dict contains 'tool' (the action dict) and 'src_id' (the parent node_id).
        In case of EVALUATION mode, the value dict contains directly the action dict.
        Transforms this into a batched sequence of steps per environment.
        """

        # print("------- ACTION COMPUTE -------")
 
        obs = self.env.unwrapped.get_obs()

        displayed_skipped_instruction = False

        for env_id, calls in tools_call.items():

            if env_id in self._eval_envs:
                stage_id = self._eval_envs[env_id].current_task_stage
                logs = self._eval_envs[env_id].logs
                stage_log_length = self._eval_envs[env_id].stage_log_start_idx
                composite_progress = self._eval_envs[env_id].composite_progress
                active_stage_error_state = getattr(self._eval_envs[env_id], "error_state", {})
            else:
                stage_id = 0
                stage_log_length = 0
                logs = []
                composite_progress = {}
                active_stage_error_state = {}
                next_stage_id = self._find_next_non_text_stage(
                    stage_id,
                    display_skipped_instruction=not displayed_skipped_instruction,
                    display_next_stage_after_skip=not displayed_skipped_instruction,
                )
                displayed_skipped_instruction = True
                if next_stage_id is None:
                    stage_id = self.task_ref.get_nb_total_stage() - 1
                else:
                    stage_id = next_stage_id
            

            tool_infos = self._compute_tool(
                task_ref=self.task_ref,
                randomizer=self.randomizer if self.randomized else None,
                calls = calls,
                env_id=env_id,
                obs=obs,
                error_state=active_stage_error_state,
                stage_id=stage_id,
                current_node_step=0,
                node_id=env_id,
                logs=logs,
                composite_progress=composite_progress,
                source_node_id=0,
                previous_tool_calls=0,
                previous_forgiven_tool_calls=0,
                original_log_length=stage_log_length,
                attributes = self.attributes
            )
            self._eval_envs[env_id] = tool_infos
            if tool_infos.is_full_error():
                # print(f"[FunctionExecutor] No poses returned for {func_name} : {tool_execution.reason}")
                continue

            self.trajectory_converter.transform_poses_in_actions(
                tool_infos,
                env_id
            )

    def step(self) -> Union[torch.Tensor, OrderedDict]:
        """
        Execute the actions in the environment.

        Returns:
            Dict: If SingleAGent env, return a batched tensor of action. For MultiAGent, return an OrderedDict
        """
        return self.trajectory_converter.step(self._eval_envs)

    def verif_ended_tool(self, obs : Dict) -> Dict[int,ToolStatus]:
        out : Dict[int,ToolStatus] = {}
        env_state_dict = None
        state_was_updated = False
        for env_id, env_infos in self._eval_envs.items():
            env_infos.compute_tool_results(obs)

            if env_infos.all_finished():
                updates = env_infos.get_state_updates()
                if updates:
                    if env_state_dict is None:
                        env_state_dict = self.env.unwrapped.get_state_dict().copy()
                    state_was_updated |= apply_env_state_updates(env_state_dict, env_id, updates)

                status = env_infos.build_tool_status()

                out[env_infos.node_id] = status
                env_infos.tool_robots = []

        # integrate attributes modification
        # if val['att_modif']:
            # if all([score != -1 for score in out]):
            #     for action, content in val['att_modif']:
            #         if action == "ADD":
            #             attributes[content[0]].append(content[1])
            #         else:
            #             attributes[content[0]].remove(content[1])
            # else:
            #     print("[TESTER] Skipping att modif due to no-stage completion")

        if state_was_updated and env_state_dict is not None:
            self.env.unwrapped.set_state_dict(env_state_dict)
            obs.clear()
            obs.update(self.env.unwrapped.get_obs())

        return self.randomizer.traduce_end(out) if self.randomized else out

    ################ private function

    def _get_logs(self, env_id: int) -> Tuple[List[Log], List[Log]]:
        """
        Return the log corresponding to the given env_id
        """
        return self._eval_envs[env_id]._get_logs()

    def _get_node_infos(self, node_id):
        """Allows to retrieve the env information (env_state and logs) from a specific nodes"""

        return {
            "logs" : self._eval_envs[node_id].logs,
            "env_state" : extract_env_state_val(self.env.unwrapped.get_state_dict().copy(),node_id),
            "composite_progress" : copy.deepcopy(self._eval_envs[node_id].composite_progress),
        }

    def print_current_stage(self):
        if 0 not in self._eval_envs:
            stage_id = self._find_next_non_text_stage(0)
        else:
            stage_id = self._eval_envs[0].current_task_stage
        if stage_id is None:
            raise RuntimeError()
        
        instruction = self.get_instruction(stage_id)
        stage = self.task_ref.stages[stage_id]

        print(f"[TESTER] Current stage: {stage_id}")
        print(f"[TESTER] Instruction: {instruction.get_content()}")
        print(f"[TESTER] Goal: {stage.stage_goal_description}")

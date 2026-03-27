# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict, List, Union, Tuple
from collections import OrderedDict
import torch
import copy

from magma_core.base.executor import ToolsBaseExecutor
from magma_core.base.data_structures import ToolInfos, Log
from magma_core.utils.global_utils import extract_env_state_val, batch_set_value

class ToolsTestingExecutor(ToolsBaseExecutor):
    """
    Tools executor is the main class responsible for transforming LLM calls into actions using specific defined class.
    """

    # Only for Evaluation Mode
    _eval_envs : Dict[int,ToolInfos]

    def __init__(
            self,
            planner_endpoint : str,
            nb_env : int = 1,
            randomized : bool = False,
        ):
        
        super().__init__(nb_env, planner_endpoint=planner_endpoint, ollama_worker=None, nb_randomization=int(randomized))

        self._eval_envs : Dict[int,ToolInfos] = {}
   
    ################ public function

    def check_env_state(self, obs : Dict):
        """
        Used for Checking if env has passed the stage in eval mode.

        Return a list of 0 (running), 1 (finish), -1 (error) representing the status of all envs.
        """
        out = []
        stage_id = self._eval_envs[0].current_task_stage
        env_ids = list(range(self.nb_env))
        env_verif = self.task_ref.verif_stage_env_completion(stage_id, obs, env_ids=env_ids).cpu().tolist()
        st = self.env.get_state_dict().copy()
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
            if stage_id != self.task_ref.get_nb_total_stage()-1:
                new_id = stage_id+1  
                self._pass_to_the_next_stage(stage_id, env_ids, st)

                while self.task_ref.is_stage_text_only(new_id):
                    print(f"[EXECUTOR] Skip Stage {new_id}")
                    new_id+=1

                situation = self.get_init_situation(new_id)
                print(f"NEXT STAGE : {new_id} with instruction {situation.instruction.get_content()}")

                for i in range(self.nb_env):
                    # Specific modifications
                    self._eval_envs[i].current_task_stage = new_id
                    self._eval_envs[i].stage_log_start_idx = len(self._eval_envs[i].logs)
                    # A new stage must rebuild its own active error profile.
                    self._eval_envs[i].error_state = {}
            else:
                print("FINISHED TASK")

        self.env.set_state_dict(st)

        return out
    
    def compute_actions(self, tools_call : Dict):
        """
        Take a batch of tools_calls. It's a dict where each key is a node_id associate with a dict.
        In case of GENERATION mode, the value dict contains 'tool' (the action dict) and 'src_id' (the parent node_id).
        In case of EVALUATION mode, the value dict contains directly the action dict.
        Transforms this into a batched sequence of steps per environment.
        """

        # print("------- ACTION COMPUTE -------")
 
        # step to have obs
        action = self.step()
        obs , _, _, _,_ = self.env.step(action)
        state_dict = self.env.get_state_dict().copy()

        for env_id, func in tools_call.items():
            func_name = func.get("name", None)
            params = func.get("arguments", {})

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

                while self.task_ref.is_stage_text_only(stage_id):
                    print(f"[EXECUTOR] Skip Stage {stage_id}")
                    stage_id+=1
            
            if func_name:
                tool_infos = self._compute_single_tool(
                    func_name,
                    params,
                    env_id,
                    obs,
                    active_stage_error_state,
                    current_node_step=0,
                    stage_id=stage_id,
                    node_id=env_id,
                    logs=logs,
                    composite_progress=composite_progress,
                    source_node_id=0,
                    original_log_length=stage_log_length
                )
            else:
                tool_infos = self._compute_multiple_tool(
                    actions = func,
                    env_id=env_id,
                    obs=obs,
                    error_state=active_stage_error_state,
                    stage_id=stage_id,
                    current_node_step=0,
                    node_id=env_id,
                    logs=logs,
                    composite_progress=composite_progress,
                    source_node_id=0,
                    original_log_length=stage_log_length
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

    def verif_ended_tool(self, obs : Dict) -> Dict:
        out = {}
        for env_id, env_infos in self._eval_envs.items():
            traj_to_compute = env_infos.compute_tool_results(obs)

            if traj_to_compute:
                self.trajectory_converter.transform_poses_in_actions(
                    env_infos,
                    env_id
                )

            if env_infos.all_finished():

                results, mess, att_modif = env_infos.build_return()
                
                # Detect planning error
                planning_error = []
                for i, r in enumerate(results):
                    if not r and env_infos.tool_robots[i].get_result() is not None:
                        planning_error.append(True)
                    else:
                        planning_error.append(False)

                out[env_infos.node_id] = {'success':results, "reason":mess, "att_modif" : att_modif, "planning_error":planning_error}
                env_infos.tool_robots = []

        return self.randomizer.traduce_end_eval(out) if self.randomized else out

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
            "env_state" : extract_env_state_val(self.env.get_state_dict().copy(),node_id),
            "composite_progress" : copy.deepcopy(self._eval_envs[node_id].composite_progress),
        }
   

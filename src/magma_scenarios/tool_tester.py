# Author : Loan BERNAT
# BSD-2-Clause

# example command:
# python3 -m magma_scenarios.tool_tester --nb_env 1 WarehouseSortingSimp

import argparse, sys, ast
from typing import Dict, Any, Optional
import threading, queue
from pathlib import Path

from .executor import ToolsTestingExecutor
from magma_core.configs import MAGMAConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a tool tester program to try your task")
    parser.add_argument(
        "task",
        type=str,
        help="Name of the task class to use (e.g., SortCubeRedCycleBluePick, PressButton)"
    )
    parser.add_argument(
        "--nb_env", '-n',
        type=int,
        default=2,
        help="The number of env to try"
    )
    parser.add_argument(
        "--randomized",
        "-r",
        action="store_true",
        help="If specified, use a randomizerWrapper"
    )
    return parser.parse_args()

def parse_cmd(line: str) -> Dict[str, Any]:
    line = line.strip()
    if not line:
        return {}

    if "(" in line and line.endswith(")"):
        name, argstr = line.split("(", 1)
        name = name.strip()
        argstr = argstr[:-1]  # drop trailing ')'

        arguments = {}
        if argstr.strip():
            for pair in argstr.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        if '|' in value:
                            value = value.replace('|',',')
                        parsed_value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        parsed_value = value
                    arguments[key] = parsed_value
                else:
                    # handle case with no '='
                    arguments[pair.strip()] = None
        return {"name": name, "arguments": arguments}
    else:
        return {"name": line, "arguments": {}}

def get_tools_name(tool_executor : ToolsTestingExecutor) -> str:
    tools = tool_executor.get_tools()
    s = "====== Tools =====\n"
    for tool in tools:
        s += f"- {tool['name']}("
        s+= ",".join(list(tool["parameters"]))
        s += ")\n"
    
    return s + "\n========\n"


attributes = None
def format_attributes(att : str) -> str:
    s= "=== Attributes ===\n"
    s+= att
    return s + "\n========\n"

def input_thread(q: queue.Queue, tool_executor : ToolsTestingExecutor):
    global attributes
    tools_names = get_tools_name(tool_executor)
    while True:
        try:
            print("\nAvailable tools:\n", tools_names, flush=True)
            print(format_attributes(str(attributes)), flush=True)
            print("> ", end="", flush=True)
            line = sys.stdin.readline().strip()
            q.put(line)
        except (EOFError, KeyboardInterrupt):
            q.put("EXIT")
            break

def resolve_config_path(path: Optional[str]) -> Optional[Path]:
    """
    Resolve config file with precedence:
    1. CLI path
    2. ./config.yaml
    3. package default
    """

    if path:
        cli_path = Path(path)
        if cli_path.exists():
            return cli_path
        raise TypeError(f"Impossible to find the config at path: {path}")

    cwd_config = Path.cwd() / "config.yaml"
    if cwd_config.exists():
        return cwd_config

    return None

def main(args):
    global attributes

    default_path = resolve_config_path(None)
    magma_config = MAGMAConfig.load(default_path)
    tool_executor = ToolsTestingExecutor(magma_config.magma_planner_address, nb_env = args.nb_env, randomized=args.randomized)
    
    env = tool_executor.initialize(args.task, {})

    cmd_queue = queue.Queue()
    
    
    obs, _ = env.reset(seed=0,options=tool_executor.get_env_options(0)) # reset with a seed for determinism
    situation = tool_executor.get_init_situation(0)
    threading.Thread(target=input_thread, args=(cmd_queue,tool_executor), daemon=True).start()
    actions = None
    stopped = []
    attributes = situation.attributes
    print("INSTRUCTION : ", situation.instruction.get_content())
    while True:
        env.render_human()

        if not actions:
            try:
                line = cmd_queue.get_nowait()
            except queue.Empty:
                line = None

            if line:
                if line == "EXIT":
                    break
                elif line.strip() == "reset":
                    obs, _ = env.reset(seed=0,options=tool_executor.get_env_options(0))
                    continue
                else:
                    cmd = parse_cmd(line)
                    print("CMD : ", cmd)
                    actions = {k: cmd.copy() for k in range(args.nb_env)}
                    tool_executor.compute_actions(
                        tools_call=actions
                    )
                    stopped = []
        
        action = tool_executor.step() # this will return the action to take for each env
        obs, _, _, _, _ = env.step(action)
        
        tools_ended = tool_executor.verif_ended_tool(obs)
        stopped.extend(tools_ended.keys())

        if len(stopped) >= args.nb_env:
            if tools_ended:
                val = next(iter(tools_ended.values()))
                print("RETURN STATUS: ", val)
                out = tool_executor.check_env_state(obs)
                print("REWARD: ", out)
                if val['att_modif']:
                    if all([score != -1 for score in out]):
                        for action, content in val['att_modif']:
                            if action == "ADD":
                                attributes[content[0]].append(content[1])
                            else:
                                attributes[content[0]].remove(content[1])
                    else:
                        print("[TESTER] Skipping att modif due to no-stage completion")
            actions = None
        
    env.close()

    return 0



if __name__ == "__main__":
    args = parse_args()
    main(args)
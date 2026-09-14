from __future__ import annotations

# Author : Loan BERNAT
# BSD-2-Clause

# example command:
# python3 -m magma_scenarios.tool_tester --nb_env 1 WarehouseSortingSimp

import argparse, ast
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .executor import ToolsTestingExecutor
import threading, queue
from pathlib import Path
from magma_scenarios import load_preset

from magma_core.configs import MAGMAConfig
from magma_core.utils.text_utils import auto_cast
from magma_core.domain import Call

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a tool tester program to try your task")
    parser.add_argument(
        "task",
        type=str,
        help="Name of the task class to use (e.g., SortCubeRedCycleBluePick, PressButton)"
    )
    parser.add_argument(
        '--nb-env', '--nb_env', '-n',
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
    args, unknown = parser.parse_known_args()

    extra_args = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown):
                value = auto_cast(unknown[i + 1])
            else:
                value = True
            extra_args[key] = value
            i += 2
        else:
            raise ValueError(f"Unexpected argument format: {unknown[i]}")

    args.extra = extra_args
    return args

def _parse_value(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        if isinstance(node, ast.Name):
            return node.id
        return ast.unparse(node)


def parse_cmd(line: str) -> list[Call]:
    line = line.strip()
    if not line:
        return []

    try:
        expr = ast.parse(line, mode="eval").body
    except SyntaxError:
        return [Call(name=line,arguments={})]

    if not isinstance(expr, ast.Call):
        return [Call(name=line,arguments={})]

    robot_name = "panda"

    if isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name):
        robot_name = expr.func.value.id
        tool_name = expr.func.attr
    elif isinstance(expr.func, ast.Name):
        tool_name = expr.func.id
    else:
        raise ValueError(f"Unsupported command syntax: {line}")

    arguments = {}
    

    for keyword in expr.keywords:
        if keyword.arg is None:
            raise ValueError("**kwargs are not supported in tool tester commands")
        arguments[keyword.arg] = _parse_value(keyword.value)

    if expr.args:
        raise ValueError("Positional arguments are not supported. Use key=value.")

    tool_call = [
        Call(
            name=tool_name,
            arguments=arguments,
            target_robot_name=robot_name
        )
    ]

    return tool_call

def get_tools_name(tool_executor : ToolsTestingExecutor) -> str:
    tools = tool_executor.get_tools()
    s = "====== Tools =====\n"
    for tool in tools:
        s += f"- {tool['name']}("
        s+= ",".join(list(tool["parameters"]))
        s += ")\n"
    
    return s + "\n========\n"


def format_attributes(att : str) -> str:
    s= "=== Attributes ===\n"
    s+= att
    return s + "\n========\n"


def input_thread(q: queue.Queue, tool_executor : ToolsTestingExecutor):
    tools_names = get_tools_name(tool_executor)
    while True:
        try:
            print("\nAvailable tools:\n", tools_names, flush=True)
            print(format_attributes(str(tool_executor.attributes)), flush=True)
            line = input("> ").strip()
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
    import torch
    from .executor import ToolsTestingExecutor

    default_path = resolve_config_path(None)
    magma_config = MAGMAConfig.load(default_path, accept_no_backend=True)
    sim_backend = magma_config.benchmark.get("sim_backend", "auto")
    if sim_backend in {"auto", "cpu"} and not torch.cuda.is_available() and args.nb_env > 1:
        print("[TESTER] CUDA unavailable, forcing --nb_env to 1 for CPU simulation.")
        args.nb_env = 1

    tool_executor = ToolsTestingExecutor(magma_config.magma_planner_address, nb_env = args.nb_env, randomized=args.randomized)
    
    Task_Cls = load_preset(args.task)
    env = tool_executor.initialize(
        Task_Cls(**args.extra),
        sim_backend=sim_backend,
    )

    cmd_queue = queue.Queue()
    
    
    obs, _ = tool_executor.reset_environment(
        tool_executor.get_env_options(),
        seed=0,
        reconfigure=False,
    )
    threading.Thread(target=input_thread, args=(cmd_queue,tool_executor), daemon=True).start()
    actions = None
    stopped = []
    print("INSTRUCTION : ", tool_executor.get_instruction(0).get_content())
    while True:
        env.unwrapped.render_human()

        if not actions:
            try:
                line = cmd_queue.get_nowait()
            except queue.Empty:
                line = None

            if line:
                if line == "EXIT":
                    break
                elif line.strip() == "reset":
                    obs, _ = tool_executor.reset_environment(
                        tool_executor.get_env_options(),
                        seed=0,
                        reconfigure=False,
                    )
                    continue
                else:
                    cmd = parse_cmd(line)
                    print("CMD : ", [c.to_string() for c in cmd])
                    actions = {k: cmd for k in range(args.nb_env)}
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
            actions = None
        
    env.close()

    return 0



if __name__ == "__main__":
    args = parse_args()
    main(args)

import time
import gymnasium as gym
import magma_scenarios.envs.packaging_env.main

env = gym.make(
    "Packaging",
    robot_init_qpos_noise=0.02,
    render_mode="human"
)

obs, info = env.reset()

print("render_mode:", env.render_mode)

for _ in range(10000):

    env.render()
    time.sleep(0.05)

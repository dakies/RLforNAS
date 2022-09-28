import gym
import numpy as np
import os
import ray
import sys
from ray.rllib.algorithms.ppo import PPO, PPOConfig

module_path = os.path.abspath(os.path.join('nas-bench-envs'))
if module_path not in sys.path:
    sys.path.append(module_path)
    os.environ['PYTHONPATH'] = module_path
from nas_bench_envs.envs.nas_bench_201_envs import NasBench201Clusters, ActionMaskEnv
from nas_bench_envs.callbacks import MetricsCallbacks
from ray.rllib.examples.models.action_mask_model import (
    TorchActionMaskModel,
)

env_config = {"cluster": 11, "network_init": 'cluster', "dataset": 'cifar10'}

config = (
    PPOConfig()
    .framework('torch')
    .resources(num_gpus=1, num_cpus_per_worker=1)
    .environment(env=NasBench201Clusters, render_env=False, env_config=env_config)
    .rollouts(horizon=1000, num_rollout_workers=8)
    .reporting()  # keep_per_episode_custom_metrics= True
    .callbacks(MetricsCallbacks)
    # .evaluation(evaluation_interval=10, evaluation_duration=1, evaluation_duration_unit="episodes",
    #             evaluation_num_workers=1,
    #             evaluation_parallel_to_training=True,
    #             evaluation_config={"explore": False, "render_env": False}, )
)

model = {
    "custom_model": TorchActionMaskModel,
    # disable action masking according to CLI
    "custom_model_config": {"no_masking": False},
}
config.training(model=model)
config.environment(env=ActionMaskEnv)

algo = PPO(config=config, env=ActionMaskEnv)
algo.restore("/home/sem22h2/ray_results/PPO_2022-09-13_09-49-20")
print("loaded agent")

# Create the env to do inference in.
env = gym.make(ActionMaskEnv(env_config))
obs = env.reset()

iterations = 1000
num_episodes = 0
episode_rewards = np.NaN(iterations)
best_reward = np.NaN(iterations)

for i in range(iterations):
    # Compute an action (`a`).
    a = algo.compute_single_action(
        observation=obs,
        explore=False,
        policy_id="default_policy",  # <- default value
    )
    # Send the computed action `a` to the env.
    obs, reward, done, _ = env.step(a)
    episode_rewards[i] = reward
    best_reward[i] = np.max(best_reward)

with open('inference.npy', 'wb') as f:
    np.save(f, episode_rewards)
    np.save(f, best_reward)

ray.shutdown()

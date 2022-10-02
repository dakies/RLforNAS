import gym
import matplotlib.pyplot as plt
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

env_config = {"cluster": 11, "network_init": 'cluster', "dataset": 'imagenet_16'}  # , "render_mode": "human"

config = (
    PPOConfig()
    .debugging(seed=123, log_level="WARN")
    .framework("torch")
    .resources(num_gpus=1, num_cpus_per_worker=1)
    .environment(env=NasBench201Clusters, render_env=False, env_config=env_config)
    .rollouts(horizon=10, num_rollout_workers=1)
    .reporting()  # keep_per_episode_custom_metrics= True
    .callbacks(MetricsCallbacks)
    .offline_data(output="logdir")
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

# Create the env to do inference in.
env = gym.make("ActionMaskEnv-v0", config=env_config)
trials = 20
steps = 50
try:
    raise ()
    with open('inference.npy', 'rb') as f:
        episode_rewards = np.load(f)
        best_reward = np.load(f)
except:
    algo = PPO(config=config, env=ActionMaskEnv)
    algo.restore(
        "/home/sem22h2/Documents/ExploringRL/results/PPO_2022-10-01_19-05-37/"
        "PPO_ActionMaskEnv_451de_00000_0_2022-10-01_19-05-37/checkpoint_000009")
    print("loaded agent")

    episode_rewards = np.zeros([trials, steps])
    best_reward = np.zeros([trials, steps])
    for trial in range(trials):
        print("Trial:{}".format(trial))
        obs = env.reset()
        for i in range(steps):
            # Compute an action (`a`).
            a = algo.compute_single_action(observation=obs,
                                           explore=False)  # policy_id="default_policy",  # <- default value
            # Send the computed action `a` to the env.
            obs, reward, done, _ = env.step(a)
            episode_rewards[trial, i] = reward
            if i == 0:
                best_reward[trial, i] = reward
            else:
                best_reward[trial, i] = np.max([best_reward[trial, i - 1], reward])
            if (i % 2) == 0:
                obs = env.reset()

    with open('inference.npy', 'wb') as f:
        np.save(f, episode_rewards)
        np.save(f, best_reward)
    ray.shutdown()
best_reward_dict = {"checkpoint_0": best_reward}

env_config = {"cluster": 11, "network_init": 'random', "dataset": 'imagenet_16'}
env = gym.make("NasBench201Clusters-v0", config=env_config)
obs = env.reset()
episode_rewards = np.zeros([trials, steps])
best_reward = np.zeros([trials, steps])
for trial in range(trials):
    for i in range(steps):
        # Compute an action (`a`).
        a = np.random.randint(env.action_space.n)
        # Send the computed action `a` to the env.
        obs, reward, done, _ = env.step(a)
        episode_rewards[trial, i] = reward
        if i == 0:
            best_reward[trial, i] = reward
        else:
            best_reward[trial, i] = np.max([best_reward[trial, i - 1], reward])
        if (i % 3) == 0:
            obs = env.reset()

with open('random_walk.npy', 'wb') as f:
    np.save(f, episode_rewards)
    np.save(f, best_reward)

best_reward_dict["random"] = best_reward

# Plot searborn
# sns.lineplot(best_reward_dict["checkpoint_0"])

# Plot matplot
steps = np.arange(best_reward_dict["random"].shape[1])
fig, ax = plt.subplots(figsize=(9, 3))
random_mean = np.mean(best_reward_dict["random"], axis=0)
ax.plot(steps, random_mean, label='Random Walk')
ci = np.std(best_reward_dict["random"], axis=0)
ax.fill_between(steps, (random_mean - ci), (random_mean + ci), color='b', alpha=.1)

random_mean = np.mean(best_reward_dict["checkpoint_0"], axis=0)
ax.plot(steps, np.mean(best_reward_dict["checkpoint_0"], axis=0), label='Reinforcement Learning')
ci = np.std(best_reward_dict["random"], axis=0)
ax.fill_between(steps, (random_mean - ci), (random_mean + ci), color='orange', alpha=.1)
ax.legend(loc='lower right')
plt.ylabel('Max network accuracy')
plt.xlabel('Iterations')
plt.title("Maximum network precision of an RL agent against a random walk")
# plt.ylim(0.7, 1)
plt.show()
plt.savefig("inference.png", bbox_inches='tight')

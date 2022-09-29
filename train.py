"""
You can visualize experiment results in ~/ray_results using TensorBoard.
Run example with defaults:
$ python custom_env.py
For CLI options:
$ python custom_env.py --help
"""
import argparse
import os
import sys

from ray.air.callbacks.wandb import WandbLoggerCallback
# from ray.rllib.agents.ppo import ppo
from ray.rllib.algorithms.ppo import PPO, PPOConfig

module_path = os.path.abspath(os.path.join('nas-bench-envs'))
if module_path not in sys.path:
    sys.path.append(module_path)
    os.environ['PYTHONPATH'] = module_path
from nas_bench_envs.envs.nas_bench_201_envs import NasBench201Clusters, ActionMaskEnv
from nas_bench_envs.callbacks import MetricsCallbacks
from ray.rllib.examples.models.action_mask_model import (
    ActionMaskModel,
    TorchActionMaskModel,
)
import ray
from ray import tune
from ray.rllib.utils.framework import try_import_torch

# tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
# )
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--network_init",
    choices=["random", "fixed", "cluster"],
    default="cluster",
    help="Initialization of the first NN architecture",
)
parser.add_argument(
    "--dataset",
    choices=["cifar10", "cifar100", "imagenet"],
    default="cifar10",
    help="Which dataset to use for accuracy of NN architectures",
)
parser.add_argument(
    "--cluster", type=int, default=11, help="Which C-BRED cluster index to aim for."
)
parser.add_argument(
    "--action_masking",
    action="store_true",
    help="Use action masking to mask invalid actions.",
)

# parser.add_argument(
#     "--as-test",
#     action="store_true",
#     help="Whether this script should be run as a test: --stop-reward must "
#     "be achieved within --stop-timesteps AND --stop-iters.",
# )
# parser.add_argument(
#     "--stop-iters", type=int, default=50, help="Number of iterations to train."
# )
# parser.add_argument(
#     "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
# )
# parser.add_argument(
#     "--stop-reward", type=float, default=0.1, help="Reward at which we stop training."
# )
# parser.add_argument(
#     "--no-tune",
#     action="store_true",
#     help="Run without Tune using a manual train loop instead. In this case,"
#     "use PPO without grid search and no TensorBoard.",
# )
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    # ModelCatalog.register_custom_model(
    #     "my_model", TorchCustomModel if args.framework == "torch" else CustomModel
    # )

    # config = {
    #     "env": SimpleCorridor,  # or "corridor" if registered above
    #     "env_config": {
    #         "corridor_length": 5,
    #     },
    #     # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    #     "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    #     "model": {
    #         "custom_model": "my_model",
    #         "vf_share_layers": True,
    #     },
    #     "num_workers": 1,  # parallelism
    #     "framework": args.framework,
    # }

    # stop = {
    #     "training_iteration": args.stop_iters,
    #     "timesteps_total": args.stop_timesteps,
    #     "episode_reward_mean": args.stop_reward,
    # }
    seed = 1234
    env_config = {"cluster": args.cluster, "network_init": args.network_init, "dataset": args.dataset, "seed": seed}
    config = (
        PPOConfig()
        .debugging(seed=seed, log_level="WARN")
        .framework(args.framework)
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
    if args.action_masking:
        model = {
            "custom_model": ActionMaskModel
            if args.framework != "torch"
            else TorchActionMaskModel,
            # disable action masking according to CLI
            "custom_model_config": {"no_masking": not args.action_masking},
        }
        config.training(model=model)
        config.environment(env=ActionMaskEnv)
    ray.init(local_mode=args.local_mode)
    results = tune.run(
        PPO,
        config=config.to_dict(),
        stop={"training_iteration": 100},
        checkpoint_freq=5,
        checkpoint_at_end=True,
        callbacks=[
            WandbLoggerCallback(api_key="c36c598399c6c7f2f0b446aac164da6c7956a263", project="RayNasBenchClustersV0")]
    )

    # print("Training completed. Restoring new Trainer for action inference.")
    # from ray.rllib.algorithms.registry import get_algorithm_class
    # import gym
    #
    # # Get the last checkpoint from the above training run.
    # checkpoint = results.get_best_result().checkpoint
    # # Create new Trainer and restore its state from the last checkpoint.
    # algo = get_algorithm_class(args.run)(config=config)
    # algo.restore(checkpoint)
    #
    # # Create the env to do inference in.
    # env = gym.make(ActionMaskEnv(env_config))
    # obs = env.reset()
    #
    # iterations = 1000
    # num_episodes = 0
    # episode_rewards = np.NaN(iterations)
    # best_reward = np.NaN(iterations)
    #
    # for i in range(iterations):
    #     # Compute an action (`a`).
    #     a = algo.compute_single_action(
    #         observation=obs,
    #         explore=args.explore_during_inference,
    #         policy_id="default_policy",  # <- default value
    #     )
    #     # Send the computed action `a` to the env.
    #     obs, reward, done, _ = env.step(a)
    #     episode_rewards[i] = reward
    #     best_reward[i] = np.max(best_reward)
    #
    # with open('inference.npy', 'wb') as f:
    #     np.save(f, episode_rewards)
    #     np.save(f, best_reward)
    #

ray.shutdown()

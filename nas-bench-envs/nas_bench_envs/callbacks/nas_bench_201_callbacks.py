from typing import Dict

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy


class MetricsCallbacks(DefaultCallbacks):
    def on_episode_start(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        episode.user_data["step_reward"] = []
        episode.user_data["in_cluster"] = []
        episode.user_data["train_time"] = []
        # episode.user_data["step_out_cluster"] = []

    def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        info = episode.last_info_for()
        if info is not None:
            episode.user_data["step_reward"].append(info["step_reward"])
            episode.user_data["in_cluster"].append(info["in_cluster"])
            episode.user_data["train_time"].append(info["train_time"])
            # episode.user_data["step_out_cluster"].append(info["step_out_cluster"])

    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            env_index: int,
            **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        episode.custom_metrics["step_reward_max"] = float(np.max(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_mean"] = float(np.mean(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_min"] = float(np.min(episode.user_data["step_reward"]))
        episode.custom_metrics["ratio_in_cluster"] = float(
            episode.user_data["in_cluster"].count(1) / len(episode.user_data["in_cluster"]))
        shift_right = np.pad(episode.user_data["in_cluster"], (1, 0), 'edge')
        step_out_cluster = np.logical_and(np.logical_not(episode.user_data["in_cluster"]), shift_right[:-1])
        episode.custom_metrics["no_step_out_cluster"] = int(np.count_nonzero(step_out_cluster == 1))
        episode.custom_metrics["train_time"] = int(np.sum(episode.user_data["train_time"]))
        # episode.custom_metrics["no_step_out_cluster"] = int(episode.user_data["step_out_cluster"].count(1))

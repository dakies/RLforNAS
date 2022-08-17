from gym.envs.registration import register

register(
    id='NasBench201-v0',
    entry_point='nas_bench_envs.envs:NasBench201',
    max_episode_steps=1000,
)
register(
    id='NasBench201Clusters-v0',
    entry_point='nas_bench_envs.envs:NasBench201Clusters',
    max_episode_steps=1000,
)

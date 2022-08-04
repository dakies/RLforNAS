from gym.envs.registration import register

register(
    id='nas_bench_envs/NasBench201Env-v0',
    entry_point='nas_bench_envs.envs:NasBench201Env',
    max_episode_steps=1000,
)

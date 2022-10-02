from ray.rllib.offline.json_reader import JsonReader

path = "/home/sem22h2/ray_results/PPO_ActionMaskEnv_2022-10-01_11-00-046berfvvb"  # /output-2022-10-01_11-01-03_worker-1_0.json

reader = JsonReader(path)
# Compute off-policy estimates
for _ in range(100):
    batch = reader.next()
    print(batch)

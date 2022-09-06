import gym
import numpy as np

from TENAS.models import CellStructure


class CbredWrapper(gym.Wrapper):
    def __init__(self, env, cluster=11):
        super().__init__(env)
        self.cluster = cluster
        # For logging
        self.in_cluster = 0
        self.in_cluster_prev = 0
        self.current_cluster = 0

    def _get_info(self):
        info = self.env._get_info()
        info.update({
            "cluster": self.current_cluster,
            "step_out_cluster": self.in_cluster == 0 and self.in_cluster_prev == 1,
            "step_in_cluster": self.in_cluster == 1 and self.in_cluster_prev == 0})
        return info

    def reset(self, **kwargs):
        v = self.vertices
        no_ops = len(self.ops)
        self.adjacency_tensor = np.zeros([no_ops, v, v])

        if self.network_init == "random":
            self.set_rand_tensor()
        elif self.network_init == "fixed":
            self.adjacency_tensor[0, 0, 3] = 1
        elif self.network_init == "cluster":
            datatable = self.datatable
            cluster = self.cluster
            cluster_idx = np.where(datatable[:, 1] == str(cluster))[0]
            arch_string = np.random.choice(datatable[cluster_idx, 0])
            genotype = arch_string
            genotype_list = CellStructure.str2fullstructure(genotype).tolist()[0]
            self.adjacency_tensor = self.get_adjacency_tensor(genotype_list)
        else:
            raise "Error: not defined initialization method."

        observation = self.env._get_obs()
        return observation

    def _get_reward(self):
        reward, cluster = self.env._get_reward()
        self.current_cluster = cluster
        self.in_cluster_prev = self.in_cluster
        if cluster != self.cluster:
            self.in_cluster = 0
            reward = reward / 2
        else:
            self.in_cluster = 1
        return reward

from typing import Optional

import gym
import numpy as np
from gym import spaces
from nats_bench import create

# Create the API instance for the topology search space in NATS
api = create("/scratch2/sem22hs2/NATS-tss-v1_0-3ffb9-simple", 'tss', fast_mode=True, verbose=False)


class NasBench201Env(gym.Env):
    metadata = {"render_modes": [], "render_fps": 1}

    def __init__(self, step_max=1000, render_mode: Optional[str] = None):
        assert render_mode is None  # or render_mode in self.metadata["render_modes"]

        # Environment definition
        self.vertices = 4
        self.ops = ['nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'skip_connect']
        self._label_to_op = {
            0: 'nor_conv_1x1',
            1: 'nor_conv_3x3',
            2: 'avg_pool_3x3',
            3: 'skip_connect',
        }

        # Current state
        v = self.vertices
        no_ops = len(self.ops)
        self.adjacency_tensor = np.zeros([no_ops, v, v])

        # Helper
        self.triu_y, self.triu_x = np.triu_indices(v, 1)  # Indices of upper triangular matrix
        self.num_triu = len(self.triu_y)  # Number of upper triangular elements in matrix
        num_triu = self.num_triu
        self.observation_space = spaces.MultiBinary(no_ops * num_triu)
        self.action_space = spaces.Discrete(no_ops * num_triu)

        # Len intervals
        self.max_episode_steps = 100

        # if self.render_mode == "human":
        #    pass
        return

    def _get_obs(self):
        tensor = self.adjacency_tensor
        obs = []
        for i in range(tensor.shape[2]):
            obs.extend(tensor[self.triu_x, self.triu_y, i])
        return obs

    def _get_info(self):
        return {"adjacency_tensor": self.adjacency_tensor}

    def _action2tensor(self, action):
        no_triu = self.num_triu
        tensor_z = int(action / no_triu)
        triu_idx = action % no_triu
        tensor_x = self.triu_x[triu_idx]
        tensor_y = self.triu_y[triu_idx]

        element = self.adjacency_tensor[tensor_z, tensor_y, tensor_x]
        new_element = not element
        if new_element is True:
            # Enforce whole row to be zero
            self.adjacency_tensor[:, tensor_y, tensor_x] = np.zeros(len(self.ops), dtype=int)
        self.adjacency_tensor[tensor_z, tensor_y, tensor_x] = new_element
        return

    def _ten2str(self):
        tensor = self.adjacency_tensor
        arch_str = "|"
        for x in range(tensor.shape[2]):  # x-axis = to
            for y in range(tensor.shape[1]):  # y-axis = from
                if x <= y:
                    continue
                if y == 0 and x != 1:
                    arch_str += "+|"

                for z in range(tensor.shape[2]):  # z axis = op
                    if not tensor[:, y, x].any():
                        arch_str += "none~" + str(y) + "|"
                        break
                    if tensor[z, y, x] == 1:
                        arch_str += self._label_to_op[z] + "~" + str(y) + "|"
        return arch_str

    def _nb201_lookup(self):
        arch_str = self._ten2str()
        # print(arch_str)
        info = api.get_more_info(arch_str, 'cifar10')
        return info

    def step(self, action):
        # Determine new adjacency matrix and observation space
        self._action2tensor(action)
        # Check matrix is upper diagonal
        for i in range(self.adjacency_tensor.shape[0]):
            matrix = self.adjacency_tensor[i, :, :]
            # breakpoint()
            assert (np.allclose(matrix, np.triu(matrix)))
        # Calculate reward
        info = self._nb201_lookup()
        reward = info["test-accuracy"] / 100
        # breakpoint()
        observation = self._get_obs()
        info = self._get_info()
        done = False
        return observation, reward, done, info

    def reset(self):
        v = self.vertices
        no_ops = len(self.ops)
        self.adjacency_tensor = np.zeros([no_ops, v, v])
        self.adjacency_tensor[0, 0, 3] = 1
        observation = self._get_obs()
        return observation

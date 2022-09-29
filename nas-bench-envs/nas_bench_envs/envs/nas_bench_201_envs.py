import gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gym import spaces
from gym.spaces import Box, Dict, Discrete
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from nats_bench import create

from TENAS.models import CellStructure

# Create the API instance for the topology search space in NATS
api = create("/scratch2/sem22hs2/NATS-tss-v1_0-3ffb9-simple", 'tss', fast_mode=True, verbose=False)


class NasBench201(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, config=None):
        # Config
        if config is None:
            config = {}
        self.network_init = config.get("network_init", "cluster")
        self.render_mode = config.get("render_mode", "rgb")
        self.dataset = config.get("dataset", "cifar10")

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

        # Environment
        self.observation_space = spaces.MultiBinary(no_ops * num_triu)
        self.action_space = spaces.Discrete(no_ops * num_triu)
        self.reward = np.nan

        # Len intervals
        self.max_episode_steps = 100

        # Load clustering labels and accuracies from C-Bred
        self.datatable = np.load('/home/sem22h2/Documents/ExploringRL/rl_dat.npy')

        # Rendering

        return

    def _get_obs(self):
        tensor = self.adjacency_tensor
        obs = []
        for i in range(tensor.shape[2]):
            obs.extend(tensor[self.triu_x, self.triu_y, i])
        return np.array(obs)

    def _get_info(self):
        return {"adjacency_tensor": self.adjacency_tensor,
                "step_reward": self.reward,
                "train_time": int(self._nb201_lookup()['train-all-time']),
                # "NATS_info": self._nb201_lookup(),
                # "cost_info": api.get_cost_info(self._ten2str(), self.dataset),
                }

    def _action2tensor(self, action):
        """
        Returns the new adjacency tensor after action has been applied.
        :param action:
        :return:
        """
        no_triu = self.num_triu
        tensor_z = int(action / no_triu)
        triu_idx = action % no_triu
        tensor_x = self.triu_x[triu_idx]
        tensor_y = self.triu_y[triu_idx]
        new_adjacency_tensor = self.adjacency_tensor.copy()
        element = self.adjacency_tensor[tensor_z, tensor_y, tensor_x]
        new_element = not element
        if new_element is True:
            # Enforce whole row to be zero
            new_adjacency_tensor[:, tensor_y, tensor_x] = np.zeros(len(self.ops), dtype=int)
        new_adjacency_tensor[tensor_z, tensor_y, tensor_x] = new_element
        return new_adjacency_tensor

    def _ten2str(self):
        """
         Convert adjacency tensor to string representation. This function might actually exist in the NATSbench utils
        """
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
        """ Get the information (accuracy, training time, etc.) from NATSBench API"""
        arch_str = self._ten2str()
        # print(arch_str)
        info = api.get_more_info(arch_str, self.dataset)
        return info

    def _get_reward(self):
        arch_str = self._ten2str()
        datatable = self.datatable
        _, cluster, acc_cifar10, acc_cifar100, acc_imagenet_16 = datatable[np.where(datatable == arch_str)[0]][0]
        cluster = int(cluster)
        # print("Cluster comparison. Current:{} Goal:{} Comparison{}".format(cluster, self.cluster,
        #                                                                    cluster != self.cluster))
        if self.dataset == 'cifar100':
            reward = float(acc_cifar100) / 100
        elif self.dataset == 'imagenet_16':
            reward = float(acc_imagenet_16) / 100
        elif self.dataset == 'cifar10':
            reward = float(acc_cifar10) / 100
        return reward, cluster

    def step(self, action):
        # Determine new adjacency matrix and observation space
        self.adjacency_tensor = self._action2tensor(action)
        # Check matrix is upper diagonal
        for i in range(self.adjacency_tensor.shape[0]):
            matrix = self.adjacency_tensor[i, :, :]
            # breakpoint()
            assert (np.allclose(matrix, np.triu(matrix)))
        # Calculate reward
        reward, _ = self._get_reward()
        self.reward = reward
        # breakpoint()
        observation = self._get_obs()
        info = self._get_info()
        done = False
        return observation, reward, done, info

    def set_rand_tensor(self):
        """
        Initialize a random upper diagonal hot-one tensor.
        :return: hot one encoded tensor
        """
        # Iterate matrix in x-y dimension
        for x, y in zip(self.triu_x, self.triu_y):
            # Chance of encoding an element in current row to 1
            if np.random.random_sample() > 0.5:
                idx_z = np.random.randint(0, self.adjacency_tensor.shape[0])
                self.adjacency_tensor[idx_z, y, x] = 1

    def get_adjacency_tensor(self, genotype_list):
        """
        Calculates the Adjacency Tensor of the given genotype, focusing both on connectivity and operations contained.
        :return: adjacency_tensor: (np.ndarray) Adjacency Tensor.
        """
        adjacency_tensor = np.zeros(
            (len(self.ops), len(genotype_list) + 1, len(genotype_list) + 1))
        for i, state in enumerate(genotype_list):
            for j, op in enumerate(state):
                source = op[1]
                if op[0] != 'none':
                    idx = np.argwhere(np.array(self.ops) == op[0])
                    adjacency_tensor[idx, source, i + 1] = 1
        return adjacency_tensor

    def reset(self, **kwargs):
        v = self.vertices
        no_ops = len(self.ops)
        self.adjacency_tensor = np.zeros([no_ops, v, v])

        if self.network_init == "random":
            self.set_rand_tensor()
        elif self.network_init == "fixed":
            self.adjacency_tensor[0, 0, 3] = 1
        else:
            raise "Error: not defined initialization method."

        observation = self._get_obs()
        return observation

    def render(self, mode="human"):
        return

    def _render_frame(self, mode: str):
        G = nx.DiGraph()
        edge_labels = {}
        for edge in zip(self.triu_x, self.triu_y):
            if self.adjacency_tensor[edge, :].any():
                G.add_edge(edge[0], edge[1])
                op = self._label_to_op[self.adjacency_tensor[edge, :].nonzero()[0][0]]
                edge_labels[edge] = op

        # explicitly set positions
        pos = {0: (0, 0), 1: (10, 15), 2: (10, -15), 3: (40, 0)}

        options = {
            # "font_size": 15,
            # "node_size": 350,
            # "node_color": "white",
            # "edgecolors": "black",
            # "linewidths": 2,
            # "width": 2,
        }
        G = G.reverse()
        nx.draw_networkx(G, pos, **options)
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels
            # font_color='red'
        )

        if self.render_mode == "human":
            # Set margins for the axes so that nodes aren't clipped
            ax = plt.gca()
            ax.margins(0.20)
            plt.axis("off")
            plt.show()

            # assert self.window is not None
            # The following line copies our drawings from `canvas` to the visible window
            # self.window.blit(canvas, canvas.get_rect())
            # pygame.event.pump()
            # pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
        else:  # rgb_array or single_rgb_array
            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.gca()

            ax.margins(0.20)
            plt.axis("off")

            canvas.draw()  # draw the canvas, cache the renderer
            return np.frombuffer(canvas.tostring_rgb(), dtype='uint8')


class NasBench201Clusters(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config=None):
        """

        :param render_mode:
        """
        # Config
        if config is None:
            config = {}
        self.network_init = config.get("network_init", "cluster")
        self.render_mode = config.get("render_mode", "rgb")
        self.dataset = config.get("dataset", "cifar10")
        self.cluster = config.get("cluster", 11)
        np.random.seed(config.get("seed", None))

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

        # Environment
        self.observation_space = spaces.MultiBinary(no_ops * self.num_triu)
        self.action_space = spaces.Discrete(no_ops * self.num_triu)
        self.reward = np.nan

        # Len intervals
        self.max_episode_steps = 100

        # Load clustering labels and accuracies from C-Bred

        self.datatable = np.load('/home/sem22h2/Documents/ExploringRL/rl_dat.npy')

        # Rendering
        if self.render_mode is not None:
            self.G = nx.DiGraph()
            plt.axis("off")
            if self.render_mode == "rgb_array":
                self.fig = Figure()
                self.canvas = FigureCanvas(self.fig)

        # For logging
        self.in_cluster = 0

        return

    def _get_obs(self):
        tensor = self.adjacency_tensor
        obs = []
        for i in range(tensor.shape[2]):
            obs.extend(tensor[self.triu_x, self.triu_y, i])
        return np.array(obs)

    def _get_info(self):
        return {"adjacency_tensor": self.adjacency_tensor,
                "step_reward": self.reward,
                "train_time": int(self._nb201_lookup()['train-all-time']),
                "in_cluster": bool(self.in_cluster),
                # "NATS_info": self._nb201_lookup(),
                # "cost_info": api.get_cost_info(self._ten2str(), self.dataset),
                }

    def _action2tensor(self, action):
        """
        Returns the new adjacency tensor after action has been applied.
        :param action:
        :return:
        """
        no_triu = self.num_triu
        tensor_z = int(action / no_triu)
        triu_idx = action % no_triu
        tensor_x = self.triu_x[triu_idx]
        tensor_y = self.triu_y[triu_idx]
        new_adjacency_tensor = self.adjacency_tensor.copy()
        element = self.adjacency_tensor[tensor_z, tensor_y, tensor_x]
        new_element = not element
        if new_element is True:
            # Enforce whole row to be zero
            new_adjacency_tensor[:, tensor_y, tensor_x] = np.zeros(len(self.ops), dtype=int)
        new_adjacency_tensor[tensor_z, tensor_y, tensor_x] = new_element
        return new_adjacency_tensor

    def _ten2str(self, adjacency_tensor):
        tensor = adjacency_tensor
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
        arch_str = self._ten2str(self.adjacency_tensor)
        # print(arch_str)
        info = api.get_more_info(arch_str, self.dataset)
        return info

    def _get_reward(self, adjacency_tensor):
        arch_str = self._ten2str(adjacency_tensor)
        datatable = self.datatable
        _, cluster, acc_cifar10, acc_cifar100, acc_imagenet_16 = datatable[np.where(datatable == arch_str)[0]][0]
        cluster = int(cluster)
        # print("Cluster comparison. Current:{} Goal:{} Comparison{}".format(cluster, self.cluster,
        #                                                                    cluster != self.cluster))
        if self.dataset == 'cifar100':
            reward = float(acc_cifar100) / 100
        elif self.dataset == 'imagenet_16':
            reward = float(acc_imagenet_16) / 100
        elif self.dataset == 'cifar10':
            reward = float(acc_cifar10) / 100
        return reward, cluster

    def set_rand_tensor(self):
        """
        Initialize a random upper diagonal hot-one tensor.
        :return: hot one encoded tensor
        """
        # Iterate matrix in x-y dimension
        for x, y in zip(self.triu_x, self.triu_y):
            # Chance of encoding an element in current row to 1
            if np.random.random_sample() > 0.5:
                idx_z = np.random.randint(0, self.adjacency_tensor.shape[0])
                self.adjacency_tensor[idx_z, y, x] = 1

    def get_adjacency_tensor(self, genotype_list):
        """
        Calculates the Adjacency Tensor of the given genotype, focusing both on connectivity and operations contained.
        :return: adjacency_tensor: (np.ndarray) Adjacency Tensor.
        """
        adjacency_tensor = np.zeros(
            (len(self.ops), len(genotype_list) + 1, len(genotype_list) + 1))
        for i, state in enumerate(genotype_list):
            for j, op in enumerate(state):
                source = op[1]
                if op[0] != 'none':
                    idx = np.argwhere(np.array(self.ops) == op[0])
                    adjacency_tensor[idx, source, i + 1] = 1
        return adjacency_tensor

    def reset(self, seed=None, options=None, **kwargs):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        v = self.vertices
        no_ops = len(self.ops)
        self.adjacency_tensor = np.zeros([no_ops, v, v])

        if self.network_init == "random":
            self.set_rand_tensor()
        elif self.network_init == "fixed":
            self.adjacency_tensor[0, 0, 3] = 1
        elif self.network_init == "cluster":
            datatable = self.datatable
            cluster_idx = np.where(datatable[:, 1] == str(self.cluster))[0]
            arch_string = np.random.choice(datatable[cluster_idx, 0])
            genotype = arch_string
            genotype_list = CellStructure.str2fullstructure(genotype).tolist()[0]
            self.adjacency_tensor = self.get_adjacency_tensor(genotype_list)
        else:
            raise "Error: not defined initialization method."

        observation = self._get_obs()
        if self.render_mode == "human":
            self._render_frame()
        _, cluster = self._get_reward(self.adjacency_tensor)
        if self.cluster != cluster:
            raise ValueError("Reseted outside target cluster. Current cluster: {}".format(cluster))
        return observation

    def step(self, action):
        # For logging interconnectivity check where all actions lead
        # action_to_cluster_counter = 0
        # for a in range(self.action_space.n):
        #     adjacency_tensor = self._action2tensor(a)
        #     reward, cluster = self._get_reward(adjacency_tensor)
        #     if cluster == self.cluster:
        #         action_to_cluster_counter += 1
        # self.action2cluster = action_to_cluster_counter

        # Determine new adjacency matrix and observation space
        new_adjacency_tensor = self._action2tensor(action)
        # Check matrix is upper diagonal
        # for i in range(self.adjacency_tensor.shape[0]):
        #     matrix = self.adjacency_tensor[i, :, :]
        #     # breakpoint()
        #     assert (np.allclose(matrix, np.triu(matrix)))

        # Calculate reward
        reward, cluster = self._get_reward(new_adjacency_tensor)
        self.reward = reward

        # Only take step if it leads to a valid cluster. Disadvantage: Agent needs to learn that there are steps it
        # cannot take
        # if cluster == self.cluster:
        if cluster != self.cluster:
            # raise Exception("Stepped outside target cluster. Current cluster: {} "
            #                 "Allowed actions: {} Stuck:{} Stepped outside target cluster"
            #                 "Prev. state: {} Current state: {}"
            #                 "".format(cluster, self.valid_actions, bool(self.stuck),
            #                           self._ten2str(self.adjacency_tensor),
            #                           self._ten2str(new_adjacency_tensor)))
            self.in_cluster = 0
            # reward = -1
        else:
            self.in_cluster = 1
        self.adjacency_tensor = new_adjacency_tensor
        observation = self._get_obs()
        info = self._get_info()
        # info["action_to_cluster_counter"] = action_to_cluster_counter
        done = False
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, done, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        G = self.G
        G.clear()
        edge_labels = {}
        for edge in zip(self.triu_x, self.triu_y):
            if self.adjacency_tensor[edge, :].any():
                G.add_edge(edge[0], edge[1])
                op = self._label_to_op[self.adjacency_tensor[edge, :].nonzero()[0][0]]
                edge_labels[edge] = op

        # explicitly set positions
        pos = {0: (0, 0), 1: (10, 15), 2: (10, -15), 3: (40, 0)}

        options = {
            # "font_size": 15,
            # "node_size": 350,
            # "node_color": "white",
            # "edgecolors": "black",
            # "linewidths": 2,
            # "width": 2,
        }
        G = G.reverse()
        nx.draw_networkx(G, pos, **options)
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels
            # font_color='red'
        )

        if self.render_mode == "human":
            # Set margins for the axes so that nodes aren't clipped
            ax = plt.gca()
            ax.margins(0.20)
            plt.axis("off")
            plt.show()
            return True
            # assert self.window is not None
            # The following line copies our drawings from `canvas` to the visible window
            # self.window.blit(canvas, canvas.get_rect())
            # pygame.event.pump()
            # pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
        else:  # rgb_array or single_rgb_array
            fig = self.fig
            canvas = self.canvas
            ax = fig.gca()
            ax.margins(0.20)
            plt.axis("off")
            canvas.draw()  # draw the canvas, cache the renderer
            rgb_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(
                fig.canvas.get_width_height()[::-1] + (3,))
            plt.clf()
            return rgb_array


class ActionMaskEnv(NasBench201Clusters):
    """An environment that publishes an action-mask each step."""

    def __init__(self, config):
        super().__init__(config)
        self._skip_env_checking = True
        # Masking only works for Discrete actions.
        assert isinstance(self.action_space, Discrete)
        # Add action_mask to observations.
        self.observation_space = Dict(
            {
                "action_mask": Box(0.0, 1.0, shape=(self.action_space.n,)),
                "observations": self.observation_space,
            }
        )
        self.stuck = False
        self.valid_actions = None

    def reset(self, **kwargs):
        obs = super().reset()
        obs = {"observations": obs}
        self._fix_action_mask(obs)

        while self.stuck:  # Reset till we reach a state we can move away from
            obs = super().reset()
            obs = {"observations": obs}
            self._fix_action_mask(obs)
        return obs

    def step(self, action):
        # Check whether action is valid.
        if not self.valid_actions[action]:
            raise ValueError(
                f"Invalid action sent to env! " f"valid_actions={self.valid_actions}"
            )
        obs, rew, _, info = super().step(action)
        obs = {"observations": obs}
        self._fix_action_mask(obs)
        if self.stuck:
            done = 1
            self.stuck = False
        else:
            done = 0
        return obs, rew, done, info

    def _fix_action_mask(self, obs):
        # Check all actions to see which ones lead into the cluster of choice.
        self.valid_actions = np.zeros(self.action_space.n)
        for action in range(self.action_space.n):
            new_adjacency_tensor = self._action2tensor(action)
            _, cluster = self._get_reward(new_adjacency_tensor)
            action_valid = cluster == self.cluster
            self.valid_actions[action] = action_valid

        if np.all(np.logical_not(self.valid_actions)):  # If non of the actions lead to the cluster
            # make all actions possible
            # self.valid_actions = np.logical_not(self.valid_actions)

            # raise Exception(
            # "No possible action that leads into cluster from state{}".format(self._ten2str(self.adjacency_tensor)))
            self.stuck = True
        else:
            self.stuck = False
        obs["action_mask"] = self.valid_actions

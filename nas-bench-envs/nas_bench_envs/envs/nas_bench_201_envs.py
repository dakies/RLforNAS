import gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gym import spaces
from gym.utils.renderer import Renderer
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from nats_bench import create

# Create the API instance for the topology search space in NATS
api = create("/scratch2/sem22hs2/NATS-tss-v1_0-3ffb9-simple", 'tss', fast_mode=True, verbose=False)


class NasBench201(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, random_init=True, render_mode=None):
        """

        :param render_mode:
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode  # Define the attribute render_mode in your environment

        # Config
        self.random_init = random_init

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

        self.renderer = Renderer(self.render_mode, self._render_frame)
        return

    def _get_obs(self):
        tensor = self.adjacency_tensor
        obs = []
        for i in range(tensor.shape[2]):
            obs.extend(tensor[self.triu_x, self.triu_y, i])
        return np.array(obs)

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

    def _set_rand_tensor(self):
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

    def reset(self):
        v = self.vertices
        no_ops = len(self.ops)
        self.adjacency_tensor = np.zeros([no_ops, v, v])

        if self.random_init:
            self._set_rand_tensor()
        else:
            self.adjacency_tensor[0, 0, 3] = 1

        observation = self._get_obs()
        return observation

    def render(self, mode="human"):
        # Just return the list of render frames collected by the Renderer.
        return self.renderer.get_renders()

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
    # Todo: make this into a wrapper of NasBench201 for clusters
    metadata = {"render_modes": [], "render_fps": 1}

    def __init__(self, random_init=True, render_mode=None, dataset='cifar_10'):
        """

        :param render_mode:
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        # Config
        self.random_init = random_init

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

        # Load clustering labels and accuracies
        self.dataset = dataset
        self.datatable = np.load('rl_dat.npy')
        self.cluster = 1

        # if self.render_mode == "human":
        #    pass
        return

    def _get_obs(self):
        tensor = self.adjacency_tensor
        obs = []
        for i in range(tensor.shape[2]):
            obs.extend(tensor[self.triu_x, self.triu_y, i])
        return np.array(obs)

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

    def _get_reward(self):
        arch_str = self._ten2str()
        datatable = self.datatable
        _, cluster, acc_cifar10, acc_cifar100, acc_imagenet_16 = datatable[np.where(datatable == arch_str)[0]][0]
        if cluster != self.cluster:
            return -1
        else:
            if self.dataset == 'cifar_100':
                return float(acc_cifar100) / 100
            elif self.dataset == 'imagenet_16':
                return float(acc_imagenet_16) / 100
            elif self.dataset == 'cifar_10':
                return float(acc_cifar10) / 100

    def step(self, action):
        # Determine new adjacency matrix and observation space
        self._action2tensor(action)
        # Check matrix is upper diagonal
        for i in range(self.adjacency_tensor.shape[0]):
            matrix = self.adjacency_tensor[i, :, :]
            # breakpoint()
            assert (np.allclose(matrix, np.triu(matrix)))
        # Calculate reward
        reward = self._get_reward()
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

    def reset(self):
        v = self.vertices
        no_ops = len(self.ops)
        self.adjacency_tensor = np.zeros([no_ops, v, v])

        if self.random_init:
            self.set_rand_tensor()
        else:
            self.adjacency_tensor[0, 0, 3] = 1

        observation = self._get_obs()
        return observation

    def render(self, mode="human"):
        # Just return the list of render frames collected by the Renderer.
        return self.renderer.get_renders()

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

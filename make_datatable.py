import numpy as np
from nats_bench import create

datatable = np.load("rl_dat.npy")
train_times = np.zeros([datatable.shape[0], 3])
api = create("/scratch2/sem22hs2/NATS-tss-v1_0-3ffb9-simple", 'tss', fast_mode=True, verbose=False)
for i in datatable.shape[0]:
    arch_string, cluster, acc_cifar10, acc_cifar100, acc_imagenet_16 = datatable[i, :]
    train_times[i, 0] = api.get_more_info(arch_string, "cifar10")['train-all-time']
    train_times[i, 1] = api.get_more_info(arch_string, "cifar100")['train-all-time']
    train_times[i, 2] = api.get_more_info(arch_string, "imagenet16")['train-all-time']

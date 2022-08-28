# Author: Mark Vero
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


# how much memory do we need to have for a forward pass
def get_computation_size(model, input_shape):
    hooks = []  # keep track of the hooks to be able to remove them
    summary = OrderedDict()

    # the following snippet is based on: https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py
    # the hook that will enable us to collect the feature-map sizes
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            layer = "%s-%i" % (class_name, module_idx + 1)
            summary[layer] = OrderedDict()

            if isinstance(output, (list, tuple)):
                summary[layer]['output_shape'] = [list(o.size())[1:] for o in output]
            else:
                summary[layer]['output_shape'] = list(output.size())[1:]

            params = 0
            if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                params += torch.prod(torch.LongTensor(list(module.weight.size()))).item()
                summary[layer][
                    'trainable'] = module.weight.requires_grad  # maybe we do not even have to store this information
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params += torch.prod(torch.LongTensor(list(module.bias.size()))).item()
            summary[layer]['nb_params'] = params

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))

    x = torch.cat((torch.rand((*input_shape)), torch.rand((*input_shape))),
                  0)  # we need at least two, to calculate the batchnorm
    if next(model.parameters()).is_cuda: x = x.cuda()  # we do the whole thing on gpu if needed

    model.apply(register_hook)
    model(x)

    # remove the hooks
    for h in hooks:
        h.remove()

    # calculate the needed memory for computation, knowing the feature_map sizes
    input_size = np.prod(np.array(input_shape)[1:])
    sizes = [input_size]  # first fm we have to store is the input
    params = [0]  # there is no param needed to get the input

    for layer in summary:
        params.append(summary[layer]['nb_params'])
        output_size = np.prod(np.array(summary[layer]['output_shape']))
        sizes.append(output_size)

    sizes = np.array(sizes)
    params = np.array(params)

    consequent_fm_sizes = sizes + np.roll(sizes, 1)
    consequent_fm_sizes[0] = sizes[0]  # the input and the output do not have to be stpred together

    nb_params = np.sum(params)

    consequent_fm_sizes_and_params = consequent_fm_sizes + params
    maxindex = np.argmax(consequent_fm_sizes_and_params)
    nb_parameters_in_cache = consequent_fm_sizes_and_params[maxindex]

    return nb_parameters_in_cache, nb_params


# a function that decides if a network fits on a given hardware
# precision in [Bytes]
# sizes in [KBytes]
def network_fits_on_hardware(model, input_shape, max_size_model=100000, max_size_computation=100000, precision=4):
    nb_parameters_in_cache, nb_params = get_computation_size(model, input_shape)
    # convert the params to the units used
    required_size_computation = precision * nb_parameters_in_cache / 1000  # in KB
    required_size_model = precision * nb_params / 1000  # in KB

    if required_size_model <= max_size_model and required_size_computation <= max_size_computation:
        print('The model fits perfectly on the hardware')
        return True
    elif required_size_model <= max_size_model and required_size_computation > max_size_computation:
        print('The intermediate feature maps are too large')
        return False
    elif required_size_model > max_size_model and required_size_computation <= max_size_computation:
        print('The model is too large')
        return False
    else:
        print('Both the model and the intermediate feature maps are too big')
        return False

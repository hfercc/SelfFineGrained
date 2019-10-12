import torch
import torch.nn as nn

from torchvision.transforms.functional import rotate

import numpy as np

def rotation(input):
    batch = input.shape[0]
    target = torch.cat([0,1,2,3] * int(batch / 4))
    target = target.long()
    for i in range(batch):
        input[i, :, :, :] = torch.rot90(input[i, :, :, :], target[i], [1, 2])

    return input, target


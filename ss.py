import torch
import torch.nn as nn

from torchvision.transforms.functional import rotate

import numpy as np

def rotation(input):
    choice = np.random.randint(4)
    batch = input.shape[0]
    target = torch.ones(batch, device = input.device) * choice
    target = target.long()
    image = torch.rot90(input, choice, [2, 3])

    return image, target


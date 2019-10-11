import torch
import torch.nn as nn

from torchvision.transforms.functional import rotate

import numpy as np

def rotation(input):
    choice = np.random.randint(4)
    batch = input.shape[0]
    target = torch.ones(batch, device = input.device) * choice
    if choice == 0:
        image = input
    elif choice == 1:
        image = rotate(input, 90)
    elif choice == 2:
        image = rotate(input, 180)
    elif choice == 3:
        image = rotate(input, 270)

    return image, target


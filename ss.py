import torch
import torch.nn as nn

from torchvision.transforms.functional import rotate

import numpy as np

def rotation(input):
    batch = input.shape[0]
    target = torch.tensor(np.random.permutation([0,1,2,3] * int(batch / 4)), device = input.device)
    target = target.long()
    image = torch.zeros_like(input)
    image.copy_(input)
    for i in range(batch):
        image[i, :, :, :] = torch.rot90(input[i, :, :, :], target[i], [1, 2])

    return image, target

class JigsawGenerator:

    def __init__(self, num_classes = 1000):
        self.permutation = np.load('permutation.npy')
        self.num_classes = num_classes

    def __call__(self, x):
        # x: list
        batch_size = x.shape[0]
        permed = []
        if batch_size > self.num_classes:
            target = list(range(self.num_classes)) * int(batch_size / self.num_classes)
            target = np.array(target)
            target = target[np.random.permutation(batch_size)]
        else:
            target = np.random.permutation(self.num_classes)[:batch_size]

        for i in range(batch_size):
            permed.append(x[i, self.permutation[target[i], :], ...].unsqueeze(0))

        permed = torch.cat(permed, 0)
        permed = permed.to(x.device)
        target = torch.from_numpy(target)
        target = target.to(x.device)
        return permed, target



def generate_permutation(num):

    perm = np.zeros((num, 16))
    for i in range(num):
        perm[i, :] = np.random.permutation(16)

    np.save('permutation.npy', perm)
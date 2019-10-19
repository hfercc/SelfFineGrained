import torch

def split_image(image, N):
    """
    image: (B, C, W, H)
    """
    batches = []

    for i in list(torch.split(image, N, dim=2)):
        batches.extend(list(torch.split(i, N, dim=3)))

    return batches

def combine_image(image, N):

    batches = []
    for i in range(N):
        batches.append(torch.cat(image[(i*N):((i + 1) * N - 1)], 3))
    return torch.cat(batches, 2)

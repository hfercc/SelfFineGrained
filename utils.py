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

    batches = torch.split(image, N * N, 1)
    print(len(batches))
    s = []
    for i in range(N):
        s.append(torch.cat(batches[(i*N):((i + 1) * N - 1)], 3))
    return torch.cat(s, 2)

import torch
import numpy as np

def split_image(image, N):
    """
    image: (B, C, W, H)
    """
    batches = []

    for i in list(torch.split(image, N, dim=2)):
        batches.extend(list(torch.split(i, N, dim=3)))

    return batches

def combine_image(image, N):
    #print(image.shape)
    batches = torch.split(image, 1, 1)
    batches = list(map(lambda x: x.squeeze(1), batches))
    
    s = []
    for i in range(N):
        s.append(torch.cat(batches[(i*N):((i + 1) * N)], 3))

    out = torch.cat(s, 2)
    return out

def get_num_split(input):
    if input.shape[-1] == 32:
        num_split = 8
    elif input.shape[-1] == 224:
        num_split = 32
    elif input.shape[-1] == 448:
        num_split = 64

    return num_split

def get_index(input):
    global args
    num_split = get_num_split(input)
    #print(input.shape)
    batches = split_image(input, num_split)
    batches = list(map(lambda x: x.unsqueeze(1), batches))
    batches = torch.cat(batches, 1) # (B, L, C, H, W)
    #print(batches.shape)
    total = batches.shape[1]
    #print(total)
    seq = np.random.permutation(total)
    t = seq[:(total // 4)]
    v = seq[(total // 4):]
    return t, v, batches

import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def set_device(cuda, device):
    if cuda is True and torch.cuda.is_available():
        torch.cuda.set_device(device=device)

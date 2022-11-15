import torch

def magnified_function(x, train_mode="only_hi", epsilon=1e-5):
    """
    torch.sign(x) * torch.pow(torch.abs(x)+eps, 0.5)
    """
    if train_mode == "only_hi":
        return x/torch.sqrt(torch.abs(x) + epsilon)
    elif train_mode == "both":
        return torch.cat((x[:,:3], x[:,3:] / torch.sqrt(torch.abs(x[:,3:]) + epsilon)), dim=1)
    else:
        return x


def demagnified_function(x, train_mode="only_hi"):
    """
    torch.sign(x) * torch.pow(torch.abs(x)+eps, 0.5)
    """
    if train_mode == "only_hi":
        return x * torch.abs(x)
    elif train_mode == "both":
        return torch.cat((x[:,:3], x[:,3:] * torch.abs(x[:,3:])), dim=1)
    else:
        return x

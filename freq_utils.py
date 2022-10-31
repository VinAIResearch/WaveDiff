import torch

def magnified_function(x, epsilon=1e-5):
    """
    torch.sign(x) * torch.pow(torch.abs(x)+eps, 0.5)
    """
    return x/torch.sqrt(torch.abs(x) + epsilon)

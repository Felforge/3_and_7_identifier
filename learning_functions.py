import torch
from torch import nn

NEURAL_NET_STRUCTURE = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)

def mnist_loss(predictions, targets):
    """Returns loss in learning to be used my model

    Args:
        predictions (torch.tensor): predicitions made my model
        targets (torch.tensor): target values of predicitions

    Returns:
        torch.tensor: Loss of the model, as a number 0 to 1, based on 
        how sure the model is of a predicition, not just a decision
    """
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()


def init_params(size, std=1.0):
    """Initialize random weights for every pixel of an image

    Args:
        size (integer/tuple): Size of resultant tensor. Put in tuple of of size and a 
        number of integers per tensor element for a weight and just an integer for a bias.
        std (float, optional): Scalar of resultant tensor. Defaults to 1.0.

    Returns:
        torch.tensor: tensor of random weights
    """
    return (torch.randn(size)*std).requires_grad_()

def batch_accuracy(xb, yb):
    """Return accuracy of predicitions as an integer in a tensor

    Args:
        xb (torch.tensor): tensor of predicitions
        yb (torch.tensor): tensor of labels to match images with 

    Returns:
        torch.tensor: Accuracy of predicitions as a tensor
    """
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

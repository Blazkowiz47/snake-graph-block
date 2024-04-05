import torch
from torch.nn import Module


class Loss(Module):
    def __init__(
        self,
    ) -> None:
        super(Loss, self).__init__()

    def forward(self, features, labels):
        b, _ = features.shape
        loss = torch.sum(-torch.log(features) * labels) / b
        return loss

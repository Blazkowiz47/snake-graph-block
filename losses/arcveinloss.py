import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F


class Loss(Module):
    def __init__(
        self,
        hdims: int,
        num_classes: int,
        lmbda: float = 0.01,
        fine_tune: bool = False,
    ) -> None:
        super(Loss, self).__init__()
        self.fine_tune = fine_tune
        self.lmbda = lmbda
        self.centroids = Parameter(torch.randn((num_classes, hdims)))

    def forward(self, features):
        _w = F.normalize(self.centroids, dim=0)
        _x = F.normalize(features, dim=1)
        cosa = torch.matmul(_x, _w)
        a = torch.acos(cosa)
        return self.lmbda * torch.sum(a)

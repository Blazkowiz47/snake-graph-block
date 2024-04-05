import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F


class Loss(Module):
    def __init__(self, feature_num, cls_num, m=0.1, s=64):
        super(Loss, self).__init__()
        self.s = s
        self.m = m
        self.W = Parameter(torch.randn(feature_num, cls_num))

    def forward(self, feature):
        _w = F.normalize(self.W, dim=0)
        _x = F.normalize(feature, dim=1)
        cosa = torch.matmul(_x, _w)
        a = torch.acos(cosa)

        top = torch.exp(torch.cos(a + self.m) * self.s)
        _top = torch.exp(torch.cos(a) * self.s)
        bottom = torch.sum(_top, dim=1, keepdim=True)

        # sina = torch.sqrt(1-torch.pow(cosa,2))
        # cosm = torch.cos(torch.tensor(self.m)).cuda()
        # sinm = torch.cos(torch.tensor(self.m)).cuda()
        # cosa_m =cosa*cosm-sina*sinm
        # top =torch.exp(cosa_m*self.s)
        # _top =torch.exp(cosa*self.s)
        # bottom =torch.sum(_top,dim=1,keepdim=True)

        return (top / (bottom - _top + top)) + 1e-10

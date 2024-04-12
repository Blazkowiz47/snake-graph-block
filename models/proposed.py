from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.nn import Linear, Softmax, Conv2d, Module, Sequential
from torch.nn.functional import adaptive_avg_pool2d

from models.config import Config
from utils.gcn_lib.torch_vertex import Grapher, GrapherConfig
from utils.dscnet.dsc_block import DSCModule


@dataclass
class ProposedBlockConfig:
    in_dim: int
    out_dim: int
    act: str
    bias: bool
    gcns: List[GrapherConfig]

    filter: int = 3
    requires_grad: bool = True
    remove_dsc: bool = False


class ProposedBlock(Module):
    def __init__(self, config: ProposedBlockConfig) -> None:
        super(ProposedBlock, self).__init__()
        self.proposed_block = Sequential(
            *[
                DSCModule(
                    config.in_dim,
                    config.out_dim,
                    config.filter,
                    stride=2,
                    bias=config.bias,
                    remove_dsc=config.remove_dsc,
                ),
                *[Grapher(c) for c in config.gcns],
                Conv2d(config.out_dim, config.out_dim, config.filter, 1, 1),
            ]
        )

    def forward(self, x):
        return self.proposed_block(x)


@dataclass
class ProposedConfig(Config):
    nclasses: int = 301
    features: int = 512
    blocks: Optional[List[ProposedBlockConfig]] = None


class ProposedNetwork(Module):
    def __init__(self, config: Config) -> None:
        super(ProposedNetwork, self).__init__()
        layers: List[Module] = []
        if not config.blocks:
            raise ValueError("Blocks not present")
        for block_config in config.blocks:
            layers.append(ProposedBlock(block_config))
        self.model = Sequential(*layers)
        self.model_init()

    def forward(self, x):
        x = self.model(x)
        x = adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def model_init(self):
        """
        Model init.
        """
        for module in self.modules():
            if isinstance(module, Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
                module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.data.zero_()
                    module.bias.requires_grad = True

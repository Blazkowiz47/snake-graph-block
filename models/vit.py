from typing import Optional
import torch
from torch.nn import (
    GELU,
    BatchNorm1d,
    Dropout,
    LayerNorm,
    Module,
    MultiheadAttention,
    Linear,
    Parameter,
    Conv2d,
    Sequential,
)


class TransformerEncoder(Module):
    def __init__(self, dims) -> None:
        super(TransformerEncoder, self).__init__()
        self.ln1 = LayerNorm(dims)
        self.k = Linear(dims, dims)
        self.q = Linear(dims, dims)
        self.v = Linear(dims, dims)
        self.ln2 = LayerNorm(dims)
        self.msa = MultiheadAttention(dims, 1)
        self.mlp = Sequential(
            Linear(dims, dims),
            GELU(),
            Linear(dims, dims),
            GELU(),
        )

    def forward(self, x):
        xl = self.ln1(x)
        k = self.k(xl)
        q = self.q(xl)
        v = self.v(xl)
        ao, _ = self.msa(q, k, v)
        ao = ao + x
        x = self.mlp(self.ln2(ao)) + ao
        return x


class VITBackbone(Module):
    def __init__(
        self,
        patch_size: int = 32,
        encoder_blocks: int = 5,
    ) -> None:
        """
        patch_size: 16 or 32
        encoder_blocks: 4
        """
        super(VITBackbone, self).__init__()
        h, w = 224, 224
        c, p, d = 1, patch_size, 197 if patch_size == 16 else 50
        n = h * w // (p * p)
        self.p = p
        self.patchencoder = Linear(p * p * c, d)
        self.pat_embed = Parameter(
            torch.zeros(  # pylint: disable=E1101
                1,
                1,
                d,
            )
        )
        self.pos_embed = Parameter(
            torch.zeros(  # pylint: disable=E1101
                1,
                n + 1,
                d,
            )
        )
        self.transformers = Sequential(
            *[TransformerEncoder(d) for _ in range(encoder_blocks)]
        )
        self.model_init()

    def forward(self, x):
        x = x.unfold(2, self.p, self.p).unfold(3, self.p, self.p)
        b, c, hp, wp, p, p = x.shape
        x = x.reshape(b, c * hp * wp, p, p)
        x = x.view(b, c * hp * wp, p * p)
        x = self.patchencoder(x)
        emb = self.pat_embed
        for _ in range(1, b):
            emb = torch.concat((emb, self.pat_embed), dim=0)
        x = torch.concat((x, emb), dim=1)
        x = x + self.pos_embed
        x = self.transformers(x)
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


class VITPredictor(Module):
    def __init__(
        self,
        n_classes,
        hdims: int = 512,
        patch_size: int = 32,
    ) -> None:
        super(VITPredictor, self).__init__()
        d = 197 if patch_size == 16 else 50
        self.regMLP = Sequential(
            BatchNorm1d(d),
            Dropout(0.25),
            Linear(d, hdims),
            GELU(),
            BatchNorm1d(hdims),
            Linear(hdims, n_classes),
        )

    def forward(self, x):
        x = x[:, 0, :]
        x = torch.squeeze(x, 1)
        x = self.regMLP(x)
        return x


class VIT(Module):
    def __init__(self, n_classes: int) -> None:
        super(VIT, self).__init__()
        self.backbone = VITBackbone()
        self.predictor = VITPredictor(n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.predictor(x)
        return x


class FineTuneVIT(Module):
    def __init__(self, n_classes: int, pretrained_path: Optional[str]) -> None:
        super(FineTuneVIT, self).__init__()
        self.backbone = VITBackbone()
        self.predictor = VITPredictor(n_classes)
        if pretrained_path:
            pretrained_model = VIT(301)
            pretrained_model.load_state_dict(torch.load(pretrained_path))
            for parameter in pretrained_model.parameters():
                parameter.requires_grad = False
            self.backbone = pretrained_model.backbone

    def forward(self, x):
        x = self.backbone(x)
        x = self.predictor(x)
        return x

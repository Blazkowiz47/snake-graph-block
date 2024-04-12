from torch.nn import Module, Softmax, GELU, BatchNorm1d, Sequential, Linear


class Predictor(Module):
    def __init__(self, classes: int, features: int = 512) -> None:
        super(Predictor, self).__init__()
        self.predictor = Sequential(
            Linear(features, 2 * features),
            BatchNorm1d(2 * features),
            GELU(),
            Linear(2 * features, classes),
            Softmax(dim=1),
        )

    def forward(self, x):
        return self.predictor(x)

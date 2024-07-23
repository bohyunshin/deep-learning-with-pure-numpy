import numpy as np

from nn.base import BaseNeuralNet
from nn.modules import Convolution
from tools.activations import MaxPooling, Sigmoid


class SingleCNN(BaseNeuralNet):
    def __init__(self, input_dim: tuple, k: int , zero_padding: int, pooling_k: int):
        super().__init__()
        self.cnn = Convolution(
            input_dim=input_dim,
            k=k,
            zero_padding=zero_padding
        )
        self.max_pooling = MaxPooling(k=pooling_k)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.cnn.forward(x)
        x = self.max_pooling.forward(x)
        x = self.sigmoid.forward(x)
        return x

if __name__ == "__main__":
    cnn = SingleCNN((4,4), 3, 1, 2)
    x = np.random.normal(1,0.5,(4,4))
    print(cnn.forward(x))
import numpy as np

from nn.base import BaseNeuralNet
from nn.modules import Convolution, Linear
from tools.activations import MaxPooling, Sigmoid, Softmax


class SingleCNN(BaseNeuralNet):
    def __init__(self, input_dim: tuple, output_dim: int, k: int , zero_padding: int, pooling_k: int):
        super().__init__()
        self.cnn = Convolution(
            input_dim=input_dim,
            k=k,
            zero_padding=zero_padding
        )
        self.max_pooling = MaxPooling(k=pooling_k)
        self.sigmoid = Sigmoid()
        self.softmax = Softmax()

        h = self.cnn.out_h - (pooling_k - 1)
        w = self.cnn.out_w - (pooling_k - 1)

        self.fc = Linear(input_dim=h*w, output_dim=output_dim)

    def forward(self, x):
        x = self.cnn.forward(x)
        x = self.max_pooling.forward(x)
        x = self.sigmoid.forward(x)
        x = self.fc.forward(x.reshape(1,-1))
        x = self.softmax.forward(x)
        return x

    def backward(self, y, pred, X):
        pass

    def step(self, lr):
        pass

    def zero_grad(self):
        pass

if __name__ == "__main__":
    cnn = SingleCNN((4,4), 3, 1, 2)
    x = np.random.normal(1,0.5,(4,4))
    print(cnn.forward(x))
from typing import Tuple

from numpy.typing import NDArray

from modules.convolution import Convolution
from modules.linear import Linear
from nn.base import BaseNeuralNet
from tools.activations import MaxPooling, Relu, Softmax


class SingleCNN(BaseNeuralNet):
    def __init__(
        self,
        input_dim: Tuple[int, int],
        output_dim: int,
        kernel_dim: tuple,
        stride: int,
        padding: str,
        pooling_size: int,
    ):
        super().__init__()
        self.cnn = Convolution(
            input_dim=input_dim, kernel_dim=kernel_dim, padding=padding, stride=stride
        )
        n, _, _ = input_dim
        self.n = n
        self.max_pooling = MaxPooling(
            input_dim=(n, self.cnn.h_out, self.cnn.w_out), k=pooling_size
        )
        self.softmax = Softmax()
        self.relu = Relu()

        h_out = self.max_pooling.h_out
        w_out = self.max_pooling.w_out
        self.fc = Linear(input_dim=h_out * w_out, output_dim=output_dim)

        self.layers = [self.cnn, self.max_pooling, self.relu, self.fc, self.softmax]
        self.gradient_step_layers += [self.cnn, self.fc]

    def forward(self, x: NDArray) -> NDArray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dx_out: NDArray) -> NDArray:
        for layer in self.layers[::-1]:
            dx_out = layer.backward(dx_out)
        return dx_out

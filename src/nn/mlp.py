from nn.base import BaseNeuralNet
from tools.activations import Relu, Softmax
from modules.linear import Linear
from loss.classification import CrossEntropyLoss
from loss.regression import MeanSquaredError


class MultipleLayerPerceptron(BaseNeuralNet):
    def __init__(self, struct, n, model="regression"):
        super().__init__()
        self.struct = struct
        self.n = n
        self.layers = []
        for i in range(1, len(struct)):
            fc = Linear(struct[i-1], struct[i])
            self.layers.append(fc)
            self.gradient_step_layers.append(fc)
            self.layers.append(Relu())
        if model == "classification":
            self.layers.append(Softmax())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dx_out):
        """
        params
        ------
        y: np.ndarray ( reg: (n,1), classification: (n,L) )
            True label.

        pred: np.ndarray ( reg: (n,1), classification: (n,L) )
            Prediction value.

        """
        # backpropagation
        for layer in self.layers[::-1]:
            dx_out = layer.backward(dx_out)
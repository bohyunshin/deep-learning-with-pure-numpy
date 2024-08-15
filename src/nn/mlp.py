import numpy as np

from nn.base import BaseNeuralNet
from tools.activations import Relu, Softmax
from nn.modules import Linear
from loss.classification import CrossEntropyLoss
from loss.regression import MeanSquaredError


class MultipleLayerPerceptronRegression:
    def __init__(self, struct, n, model="regression"):
        super().__init__()
        self.struct = struct
        self.n = n

        if model == "regression":
            self.loss = MeanSquaredError()
        elif model == "classification":
            self.loss = CrossEntropyLoss()
        else:
            raise

        self.model = model
        self.layers = []
        self.gradient_step_layers = []
        for i in range(1, len(struct)):
            fc = Linear(struct[i-1], struct[i])
            self.layers.append(fc)
            self.gradient_step_layers.append(fc)
            self.layers.append(Relu())
        if self.model == "classification":
            self.layers.append(Softmax())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y, pred):
        """
        params
        ------
        y: np.ndarray ( reg: (n,1), classification: (n,L) )
            True label.

        pred: np.ndarray ( reg: (n,1), classification: (n,L) )
            Prediction value.

        """
        # calculate initial gradient w.r.t. loss function
        dx = self.loss.backward(y, pred)

        # backpropagation
        for layer in self.layers[::-1]:
            dx = layer.backward(dx)

    def step(self, lr):
        for layer in self.gradient_step_layers:
            params_info = layer.get_params_grad()
            for param,info in params_info.items():
                param_grad_step = info["current"] - lr*info["grad"]
                setattr(layer, param, param_grad_step)


class MultipleLayerPerceptronClassification:
    def __init__(self, struct, n):
        super().__init__()
        self.struct = struct
        self.n = n
        self.layers = []
        self.gradient_step_layers = []
        for i in range(1, len(struct)):
            fc = Linear(struct[i-1], struct[i])
            self.layers.append(fc)
            self.gradient_step_layers.append(fc)
            self.layers.append(Relu())
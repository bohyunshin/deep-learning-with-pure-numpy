import numpy as np

from nn.base import BaseNeuralNet
from tools.activations import Relu
from nn.modules import Linear

np.random.seed(1)


class MultipleLayerPerceptron:
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
            # if i != len(struct)-1:
            #     # self.layers.append( Sigmoid() )
            #     self.layers.append( Relu() )

    def forward(self, x):
        self.activated_val = [x]
        for layer in self.layers:
            x = layer.forward(x)
            if layer.__class__.__name__ == "Sigmoid":
                self.activated_val.append(x)
        return x

    def backward(self, y, pred):
        dx = (pred - y) / self.n * 2
        for layer in self.layers[::-1]:
            dx = layer.backward(dx)

    def step(self, lr):
        for layer in self.gradient_step_layers:
            params_info = layer.get_params_grad()
            for param,info in params_info.items():
                param_grad_step = info["current"] - lr*info["grad"]
                setattr(layer, param, param_grad_step)

    def zero_grad(self):
        for layer in self.layers:
            if layer.__class__.__name__ == "Linear":
                layer.weight_grad = None
                layer.bias_grad = None

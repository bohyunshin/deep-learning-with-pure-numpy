from abc import abstractmethod


class BaseNeuralNet:
    def __init__(self):
        self.gradient_step_layers = []

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dx_out):
        pass

    def step(self, lr):
        for layer in self.gradient_step_layers:
            params_info = layer.get_params_grad()
            for param, info in params_info.items():
                param_grad_step = info["current"] - lr * info["grad"]
                setattr(layer, param, param_grad_step)

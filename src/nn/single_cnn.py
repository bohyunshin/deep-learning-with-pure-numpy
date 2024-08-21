from nn.base import BaseNeuralNet
from nn.modules import Convolution, Linear
from tools.activations import MaxPooling, Softmax, Relu


class SingleCNN(BaseNeuralNet):
    def __init__(self, input_dim, output_dim: int, kernel_dim: tuple , padding: str, pooling_size: int):
        super().__init__()
        self.cnn = Convolution(
            input_dim=input_dim,
            kernel_dim=kernel_dim,
            padding = padding
        )
        n, _, _ = input_dim
        self.n = n
        self.max_pooling = MaxPooling(input_dim=(n, self.cnn.h_out, self.cnn.w_out),
                                      k=pooling_size)
        self.softmax = Softmax()
        self.relu = Relu()

        h_out = self.max_pooling.h_out
        w_out = self.max_pooling.w_out
        self.fc = Linear(input_dim=h_out*w_out, output_dim=output_dim)

        self.gradient_step_layers = [self.cnn, self.fc]

    def forward(self, x):
        n, h_in, w_in = x.shape
        x = self.cnn.forward(x) # (n, h_out, w_out)
        x = self.max_pooling.forward(x) # (n, h_out // k, w_out // k)
        x = self.relu.forward(x)
        x = self.fc.forward(x.reshape(n,-1)) # (n, h_out*w_out) -> (n, out_dim)
        x = self.softmax.forward(x)
        return x

    def backward(self, dx_out):
        """
        params
        ------
        dx_out: np.ndarray (n, n_label)
            Upstream gradient from loss function
        """
        dx_out = self.softmax.backward(dx_out) # (n, n_label)
        dx_out = self.fc.backward(dx_out) # (n, h_in)
        dx_out = self.relu.backward(dx_out)
        dx_out = self.max_pooling.backward(dx_out.reshape(self.n, self.max_pooling.h_out, -1)) # (n, h_in, w_in)
        dx_out = self.cnn.backward(dx_out) # (n, h_in, w_in)
        return dx_out

    def step(self, lr):
        for layer in self.gradient_step_layers:
            params_info = layer.get_params_grad()
            for param,info in params_info.items():
                param_grad_step = info["current"] - lr*info["grad"]
                setattr(layer, param, param_grad_step)

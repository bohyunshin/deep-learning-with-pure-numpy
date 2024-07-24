import numpy as np


class Linear:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = np.random.normal(0, 0.5, (input_dim, output_dim))
        self.bias = np.random.normal(0, 0.5, output_dim)

        self.weight_grad = None
        self.bias_grad = None

    def forward(self, x):
        # shape of x: (num_of_data, input_dim)
        assert x.shape[1] == self.input_dim
        x = np.dot(x, self.weight) + self.bias.reshape(1, self.output_dim)
        return x # shape: (num_of_data, output_dim)


class Convolution:
    def __init__(self, input_dim: tuple, k: int , zero_padding: int):
        self.input_dim = input_dim
        h,w = input_dim
        self.out_h = (h + 2 * zero_padding - k + 1)
        self.out_w = (w + 2 * zero_padding - k + 1)
        self.kernel = np.random.normal(0, 0.5, (k, k))
        self.bias = np.random.normal(0, 0.5, (1))
        self.zero_padding = zero_padding

        self.kernel_grad = None
        self.bias_grad = None

    def forward(self, x):
        x = np.pad(x, self.zero_padding)
        h,w = x.shape
        k = self.kernel.shape[0]
        res = []
        print(h-k+1)
        for i in range(h-k+1):
            for j in range(w-k+1):
                res.append( (x[i:i+k,j:j+k] * self.kernel).sum() + self.bias )
        return np.array(res).reshape((self.out_h, self.out_w))

if __name__ == "__main__":
    input_dim = (4,4)
    kernel = np.array([[1,0,1], [0,1,0], [1,0,1]])
    padding = 1
    k = 3
    conv = Convolution(input_dim, k, padding)
    x = np.array([
        [1,0,2,0],
        [0,3,0,4],
        [5,0,6,0],
        [0,7,0,8]
    ])
    print(conv.forward(x))
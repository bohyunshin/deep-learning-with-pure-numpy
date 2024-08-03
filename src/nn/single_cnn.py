import numpy as np

from nn.base import BaseNeuralNet
from nn.modules import Convolution, Linear
from tools.activations import MaxPooling, Sigmoid, Softmax
from loss.classification import cross_entropy


class SingleCNN(BaseNeuralNet):
    def __init__(self, input_dim, output_dim: int, kernel_dim: tuple , padding: str, pooling_size: int):
        super().__init__()
        self.cnn = Convolution(
            input_dim=input_dim,
            kernel_dim=kernel_dim,
            padding = padding
        )
        n, _, _ = input_dim
        self.max_pooling = MaxPooling(input_dim=(n, self.cnn.h_out, self.cnn.w_out),
                                      k=pooling_size)
        # self.sigmoid = Sigmoid()
        self.softmax = Softmax()

        # h = self.cnn.out_h - (pooling_size - 1)
        # w = self.cnn.out_w - (pooling_size - 1)

        h_out = self.max_pooling.h_out
        w_out = self.max_pooling.w_out
        self.fc = Linear(input_dim=h_out*w_out, output_dim=output_dim)

    def forward(self, x):
        n, h_in, w_in = x.shape
        x = self.cnn.forward(x) # (n, h_out, w_out)
        x = self.max_pooling.forward(x) # (n, h_out // k, w_out // k)
        # x = self.sigmoid.forward(x)
        x = self.fc.forward(x.reshape(n,-1)) # (n, h_out*w_out) -> (n, out_dim)
        x = self.softmax.forward(x)
        return x

    def backward(self, y, pred, X):
        pass

    def step(self, lr):
        pass

    def zero_grad(self):
        pass

if __name__ == "__main__":
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml('mnist_784', as_frame=False)
    X = [x.reshape(28, -1) for x in mnist.data]
    h = X[0].shape[0]
    target = [int(i) for i in mnist.target]
    num_label = len(np.unique(target))
    one_hot_target = np.eye(num_label)[target]
    epoch = 10

    imgs = np.random.normal(0, 0.5, (100, 15, 15))
    output_dim = 9
    kernel_dim = (3,3)
    padding = "same"
    pooling_size = 3

    cnn = SingleCNN(input_dim=imgs.shape,
                    output_dim=output_dim,
                    kernel_dim=kernel_dim,
                    padding=padding,
                    pooling_size=pooling_size)

    print(cnn.forward(imgs).shape)

    # for i in range(epoch):
    #
    #     loss = 0
    #     correct = 0
    #
    #     for x,y in zip(X, one_hot_target):
    #         y_prob_pred = cnn.forward(x)
    #         y_pred = y_prob_pred.argmax()
    #
    #         correct += (y_pred == y.argmax())
    #         loss += cross_entropy(y, y_prob_pred)
    #
    #     print(f"epoch {epoch}: {round(loss, 4)} loss / {round(correct / len(X), 4)} accuracy")
    #
    #     break
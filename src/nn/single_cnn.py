import numpy as np

from nn.base import BaseNeuralNet
from nn.modules import Convolution, Linear
from tools.activations import MaxPooling, Sigmoid, Softmax
from loss.classification import cross_entropy


class SingleCNN(BaseNeuralNet):
    def __init__(self, input_dim: tuple, output_dim: int, kernel_size: int , zero_padding: int, pooling_size: int):
        super().__init__()
        self.cnn = Convolution(
            input_dim=input_dim,
            k=kernel_size,
            zero_padding=zero_padding
        )
        self.max_pooling = MaxPooling(k=pooling_size)
        self.sigmoid = Sigmoid()
        self.softmax = Softmax()

        h = self.cnn.out_h - (pooling_size - 1)
        w = self.cnn.out_w - (pooling_size - 1)

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
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml('mnist_784', as_frame=False)
    X = [x.reshape(28, -1) for x in mnist.data]
    h = X[0].shape[0]
    target = [int(i) for i in mnist.target]
    num_label = len(np.unique(target))
    one_hot_target = np.eye(num_label)[target]
    epoch = 10

    cnn = SingleCNN((h, h), num_label, 3, 1, 2)

    for i in range(epoch):

        loss = 0
        correct = 0

        for x,y in zip(X, one_hot_target):
            y_prob_pred = cnn.forward(x)
            y_pred = y_prob_pred.argmax()

            correct += (y_pred == y.argmax())
            loss += cross_entropy(y, y_prob_pred)

        print(f"epoch {epoch}: {round(loss, 4)} loss / {round(correct / len(X), 4)} accuracy")

        break
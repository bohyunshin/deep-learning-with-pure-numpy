import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nn.mlp import NeuralNetwork
from loss.regression import mean_squared_error

def train(epoch):
    # n = 1000
    # k = 4
    # struct = [4, 3, 2, 1]
    # X = np.random.randn(n, k)
    # y = np.cos(X).sum(axis=1).reshape(n,1)

    struct = [1, 5, 5, 5, 1]

    n=100
    X = np.linspace(-5, 5, n)
    y = np.sin(X)

    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    nn = NeuralNetwork(struct=struct, n=n)
    loss_ = []
    for _ in range(epoch):

        # forward pass
        pred = nn.forward(X)

        # calculate loss
        loss = mean_squared_error(y, pred)
        loss_.append(loss)

        # backward
        nn.backward(y, pred, X)

        # gradient descent
        nn.step(0.001)

        # clear gradient
        nn.zero_grad()

    sns.lineplot(loss_)
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.show()

if __name__ == "__main__":
    train(50000)
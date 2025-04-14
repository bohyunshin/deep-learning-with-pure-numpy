import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from loss.regression import mean_squared_error
from nn.mlp import MultipleLayerPerceptronRegression


def train(epoch):
    n = 1000
    k = 4
    struct = [4, 3, 2, 1]
    struct = [4, 1]
    np.random.seed(1)
    X = np.random.randn(n, k)
    y = np.cos(X).sum(axis=1).reshape(n, 1)

    nn = MultipleLayerPerceptronRegression(struct=struct, n=n)
    loss_ = []
    for _ in range(epoch):
        # forward pass
        pred = nn.forward(X)

        # calculate loss
        loss = mean_squared_error(y, pred)
        print(loss)
        loss_.append(loss)

        # backward
        nn.backward(y, pred)

        # gradient descent
        nn.step(0.001)

        # clear gradient
        # nn.zero_grad()

    sns.lineplot(loss_)
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.show()


if __name__ == "__main__":
    train(5000)

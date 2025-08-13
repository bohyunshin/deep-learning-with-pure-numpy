import numpy as np

from loss.regression import MeanSquaredError
from nn.mlp import MultipleLayerPerceptron


def main():
    n = 1000
    k = 4
    struct = [4, 3, 2, 1]

    # data generation
    np.random.seed(1)
    X_train = np.random.randn(n, k)
    y_train = np.cos(X_train).sum(axis=1).reshape(n, 1)
    epochs = 20
    lr = 0.01

    mlp = MultipleLayerPerceptron(struct=struct, n=n, model="regression")
    mse_loss = MeanSquaredError()
    losses = []

    for epoch in range(epochs):
        # calculate prediction
        y_pred = mlp.forward(X_train)

        # calculate loss
        loss = mse_loss.forward(y_train, y_pred)

        # start backpropagation
        # -> batch gradient descent using full data
        dx_out = mse_loss.backward(y_train, y_pred)
        mlp.backward(dx_out)
        mlp.step(lr)

        losses.append(loss)
        print(f"epoch {epoch} loss: {round(loss, 5)}")


if __name__ == "__main__":
    main()

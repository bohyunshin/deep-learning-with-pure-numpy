import numpy as np
from sklearn.datasets import make_gaussian_quantiles

from loss.classification import CrossEntropyLoss
from nn.mlp import MultipleLayerPerceptron
from tools.utils import one_hot_vector


def main():
    n = 1000
    k = 4
    L = 3
    struct = [4, L]

    # data generation
    np.random.seed(1)
    X_train, y_train = make_gaussian_quantiles(
        cov=3.0, n_samples=n, n_features=k, n_classes=L, random_state=1
    )
    y_train = one_hot_vector(L, y_train)
    epochs = 20
    lr = 0.01

    mlp = MultipleLayerPerceptron(struct=struct, n=n, model="classification")
    ce = CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        # calculate prediction
        y_pred = mlp.forward(X_train)

        # calculate loss
        loss = ce.forward(y_train, y_pred)

        # start backpropagation
        # -> batch gradient descent using full data
        dx_out = ce.backward(y_train, y_pred)
        mlp.backward(dx_out)
        mlp.step(lr)

        losses.append(loss)
        print(f"epoch {epoch} loss: {round(loss, 5)}")


if __name__ == "__main__":
    main()

import numpy as np
from src.nn.single_cnn import SingleCNN
from src.loss.classification import cross_entropy


def test_single_cnn_dummy_data():
    imgs = np.random.normal(0, 0.5, (100, 15, 15))
    output_dim = 9
    y = np.eye(output_dim)[np.random.choice(output_dim, 100)]
    kernel_dim = (3,3)
    padding = "same"
    pooling_size = 3
    epoch = 30

    cnn = SingleCNN(input_dim=imgs.shape,
                    output_dim=output_dim,
                    kernel_dim=kernel_dim,
                    padding=padding,
                    pooling_size=pooling_size)

    for i in range(epoch):

        y_pred_prob = cnn.forward(imgs)
        y_pred = y_pred_prob.argmax(axis=1)
        cnn.backward(y, y_pred_prob, imgs)
        cnn.step(0.01)

        loss = cross_entropy(y, y_pred_prob)
        correct = (y_pred == y.argmax(axis=1)).sum()

        print(f"epoch: {i} / loss: {loss} / accuracy: {correct / 100 * 100}%")


def test_single_cnn_mnist_data():
    pass
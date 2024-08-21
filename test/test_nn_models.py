import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))
sys.path.append(os.path.join(os.getcwd(), "test"))

from nn.single_cnn import SingleCNN
from nn.mlp import MultipleLayerPerceptron
from loss.classification import CrossEntropyLoss
from loss.regression import MeanSquaredError
from torch_model import TorchMLP
from tools.utils import one_hot_vector


# torch의 Dataset 을 상속.
class TensorData(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


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
    ce_loss = CrossEntropyLoss()

    for i in range(epoch):

        y_pred_prob = cnn.forward(imgs)
        y_pred = y_pred_prob.argmax(axis=1)
        cnn.backward(y, y_pred_prob, imgs)
        cnn.step(0.01)

        loss = ce_loss.forward(y, y_pred_prob)
        correct = (y_pred == y.argmax(axis=1)).sum()

        print(f"epoch: {i} / loss: {loss} / accuracy: {correct / 100 * 100}%")


def test_single_cnn_mnist_data():
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml('mnist_784', as_frame=False)
    n, r_c = mnist.data.shape
    X = mnist.data.reshape(n,int(np.sqrt(r_c)),-1)
    target = [int(i) for i in mnist.target]
    num_label = len(np.unique(target))
    y = np.eye(num_label)[target]

    kernel_dim = (3, 3)
    padding = "same"
    pooling_size = 3
    epoch = 30

    cnn = SingleCNN(input_dim=X.shape,
                    output_dim=num_label,
                    kernel_dim=kernel_dim,
                    padding=padding,
                    pooling_size=pooling_size)
    ce_loss = CrossEntropyLoss()

    for i in range(epoch):
        y_pred_prob = cnn.forward(X)
        y_pred = y_pred_prob.argmax(axis=1)
        cnn.backward(y, y_pred_prob, X)
        cnn.step(0.01)

        loss = ce_loss.forward(y, y_pred_prob)
        correct = (y_pred == y.argmax(axis=1)).sum()

        print(f"epoch: {i} / loss: {loss} / accuracy: {correct / 100 * 100}%")


def test_mlp_reg_same_as_torch():
    n = 1000
    k = 4
    struct = [4, 1]

    # data generation
    np.random.seed(1)
    X = np.random.randn(n, k)
    y = np.cos(X).sum(axis=1).reshape(n, 1)
    epoch = 1

    # weight, bias generation
    np.random.seed(1)
    weight = np.random.uniform(-0.1, 0.1, (1, 4))
    bias = np.random.uniform(-0.1, 0.1, 1)

    ###### numpy implementation ######
    mlp = MultipleLayerPerceptron(struct=struct, n=n, model="regression")
    mlp.layers[0].w = weight.T
    mlp.layers[0].b = bias
    loss_ = []
    mse_loss = MeanSquaredError()
    for _ in range(epoch):
        # forward pass
        pred = mlp.forward(X)

        # calculate loss
        loss = mse_loss.forward(y, pred)
        loss_.append(loss)

        # backward
        mlp.backward(y, pred)

        # gradient descent
        mlp.step(0.001)

    ###### torch implementation ######
    trainsets = TensorData(X, y)
    trainloader = torch.utils.data.DataLoader(trainsets, batch_size=n) # assuming batch gradient descent
    model = TorchMLP(struct[0], struct[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.fc1.weight.data = torch.tensor(torch.from_numpy(weight), dtype=torch.float32)
    model.fc1.bias.data = torch.tensor(torch.from_numpy(bias), dtype=torch.float32)

    loss_ = []
    n_batch = len(trainloader)

    for _ in range(epoch):

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):

            X_train, y_true = data

            optimizer.zero_grad()

            y_pred = model(X_train)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss_.append(running_loss / n_batch)

    np.testing.assert_array_almost_equal(mlp.layers[0].dw, model.fc1.weight.grad.numpy().T, decimal=6)
    np.testing.assert_array_almost_equal(mlp.layers[0].db, model.fc1.bias.grad.numpy(), decimal=6)


def test_mlp_classification_same_as_torch():
    n = 1000
    k = 4
    L = 3
    struct = [4, 3]

    # data generation
    from sklearn.datasets import make_gaussian_quantiles
    X, y = make_gaussian_quantiles(cov=3.,
                                     n_samples=n, n_features=k,
                                     n_classes=L, random_state=1)
    y = one_hot_vector(L, y)
    epoch = 1
    lr = 0.01

    # weight, bias generation
    np.random.seed(1)
    weight = np.random.uniform(-0.1, 0.1, (3, 4)) # for torch, (output_dim, input_dim)
    bias = np.random.uniform(-0.1, 0.1, 1)

    ###### numpy implementation ######

    mlp = MultipleLayerPerceptron(struct=struct, n=n, model="classification")

    # set weight, bias same as torch model
    mlp.layers[0].w = weight.T # (input_dim, output_dim)
    mlp.layers[0].b = bias

    loss_ = []
    for _ in range(epoch):
        # forward pass
        prob_pred = mlp.forward(X) # not logit, probability prediction

        # calculate loss
        loss = mlp.loss.forward(y, prob_pred)
        loss_.append(loss)

        # backward
        mlp.backward(y, prob_pred)

        # gradient descent
        mlp.step(lr)
    np_loss = loss

    ###### torch implementation ######
    trainsets = TensorData(X, y)
    trainloader = torch.utils.data.DataLoader(trainsets, batch_size=n)  # assuming batch gradient descent
    model = TorchMLP(struct[0], struct[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.fc.weight.data = torch.tensor(torch.from_numpy(weight), dtype=torch.float32)
    model.fc.bias.data = torch.tensor(torch.from_numpy(bias), dtype=torch.float32)

    loss_ = []
    n_batch = len(trainloader) # batch gradient descent (not mini-batch)

    for _ in range(epoch):

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            X_train, y_true = data

            optimizer.zero_grad()

            y_pred = model(X_train) # this is logit, not probability prediction
            loss = criterion(y_pred, y_true) # logit should be input for nn.CrossEntropyLoss()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss_.append(running_loss / n_batch)
    torch_loss = loss.detach().numpy()

    np.testing.assert_array_almost_equal(np_loss, torch_loss)
    np.testing.assert_array_almost_equal(model.fc.bias.detach().numpy(), mlp.layers[0].b)
    np.testing.assert_array_almost_equal(model.fc.weight.detach().numpy(), mlp.layers[0].w.T)
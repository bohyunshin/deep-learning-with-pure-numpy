import os
import sys

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.getcwd(), "src"))
sys.path.append(os.path.join(os.getcwd(), "test"))

from torch_model import TorchCNN, TorchMLP

from data.data import NumpyDataLoader, NumpyDataset
from loss.classification import CrossEntropyLoss
from loss.regression import MeanSquaredError
from nn.mlp import MultipleLayerPerceptron
from nn.single_cnn import SingleCNN
from tools.utils import one_hot_vector

torch.set_default_dtype(torch.float64)


class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def test_single_cnn_dummy_data_same_as_torch():
    n = 1000
    h_in, w_in = 15, 15
    n_channel = 1
    imgs = torch.randn((n, n_channel, h_in, w_in))
    output_dim = 9
    y = np.eye(output_dim)[np.random.choice(output_dim, n)]
    kernel_dim = (3, 3)
    padding = "same"
    pooling_size = 3
    epoch = 1
    lr = 0.1
    batch_size = 32
    n_batch = n // batch_size + (n % batch_size >= 1)

    # use numpy dataloader
    dataset = NumpyDataset(imgs, y)
    dataloader = NumpyDataLoader(dataset, batch_size=batch_size, shuffle=False)

    # set numpy model
    cnn = SingleCNN(
        input_dim=(n, h_in, w_in),
        output_dim=output_dim,
        kernel_dim=kernel_dim,
        padding=padding,
        pooling_size=pooling_size,
    )
    ce_loss = CrossEntropyLoss()

    # set torch model
    model = TorchCNN(h_in, w_in, output_dim, kernel_dim, pooling_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # set numpy model weight as torch model weight in advance
    cnn.cnn.kernel = model.conv.weight.data.squeeze(0, 1).detach().numpy().copy()
    cnn.cnn.b = model.conv.bias.data.detach().numpy().copy()
    cnn.fc.w = model.fc.weight.data.detach().numpy().copy().T
    cnn.fc.b = model.fc.bias.data.detach().numpy().copy()

    for _ in range(epoch):
        running_loss_np = 0.0
        running_loss_pt = 0.0

        for data in dataloader:
            X_train, y_train = data
            y_train = torch.tensor(torch.from_numpy(y_train.copy()))

            # torch implementation
            optimizer.zero_grad()
            y_pred = model(X_train)
            y_pred.retain_grad()
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            running_loss_pt += loss.item()

            # numpy implementation
            X_train = X_train.squeeze(1).detach().numpy()  # no channel
            y_train = y_train.detach().numpy()
            y_pred_prob = cnn.forward(X_train)  # not logits, probability prediction
            loss = ce_loss.forward(y_train, y_pred_prob)
            running_loss_np += loss.item()

            dx_out = ce_loss.backward(y_train, y_pred_prob)
            cnn.backward(dx_out)
            cnn.step(lr)

        running_loss_pt /= n_batch
        running_loss_np /= n_batch

        # check loss at every epoch
        np.testing.assert_almost_equal(running_loss_pt, running_loss_np)

    # fc layer check
    np.testing.assert_array_almost_equal(model.fc.bias.detach().numpy(), cnn.fc.b)
    np.testing.assert_array_almost_equal(model.fc.weight.detach().numpy(), cnn.fc.w.T)
    # conv layer check
    np.testing.assert_array_almost_equal(
        model.conv.weight.squeeze((0, 1)).detach().numpy(), cnn.cnn.kernel
    )
    np.testing.assert_array_almost_equal(model.conv.bias.detach().numpy(), cnn.cnn.b)


def test_mlp_reg_same_as_torch():
    n = 1000
    k = 4
    struct = [4, 1]

    # data generation
    np.random.seed(1)
    X = np.random.randn(n, k)
    y = np.cos(X).sum(axis=1).reshape(n, 1)
    epoch = 1
    batch_size = 32
    n_batch = n // batch_size + (n % batch_size >= 1)
    lr = 0.01

    # use numpy dataloader
    dataset = NumpyDataset(X, y)
    dataloader = NumpyDataLoader(dataset, batch_size=batch_size, shuffle=False)

    # set torch model
    model = TorchMLP(struct[0], struct[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # set numpy model
    mlp = MultipleLayerPerceptron(struct=struct, n=n, model="regression")
    mse_loss = MeanSquaredError()

    # set numpy model weight as torch model weight in advance
    mlp.layers[0].w = model.fc.weight.T.detach().numpy().copy()
    mlp.layers[0].b = model.fc.bias.detach().numpy().copy()

    for _ in range(epoch):
        running_loss_np = 0.0
        running_loss_pt = 0.0

        for data in dataloader:
            X_train, y_train = data

            X_train = torch.tensor(torch.from_numpy(X_train.copy()))
            y_train = torch.tensor(torch.from_numpy(y_train.copy()))

            # torch implementation
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss_pt = criterion(y_pred, y_train)
            loss_pt.backward()
            optimizer.step()
            running_loss_pt += loss_pt.item()

            # numpy implementation
            X_train = X_train.detach().numpy()
            y_train = y_train.detach().numpy()
            y_pred = mlp.forward(X_train)
            loss_np = mse_loss.forward(y_train, y_pred)
            dx_out = mse_loss.backward(y_train, y_pred)
            mlp.backward(dx_out)
            mlp.step(lr)
            running_loss_np += loss_np.item()

        running_loss_pt /= n_batch
        running_loss_np /= n_batch

        # check loss at every epoch
        np.testing.assert_almost_equal(running_loss_pt, running_loss_np)

    np.testing.assert_array_almost_equal(
        mlp.layers[0].w, model.fc.weight.T.detach().numpy()
    )
    np.testing.assert_array_almost_equal(
        mlp.layers[0].b, model.fc.bias.detach().numpy()
    )


def test_mlp_classification_same_as_torch():
    n = 1000
    k = 4
    L = 3
    struct = [4, 3]

    # data generation
    from sklearn.datasets import make_gaussian_quantiles

    X, y = make_gaussian_quantiles(
        cov=3.0, n_samples=n, n_features=k, n_classes=L, random_state=1
    )
    epoch = 1
    batch_size = 32
    n_batch = n // batch_size + (n % batch_size >= 1)
    lr = 0.01

    # use numpy dataloader
    dataset = NumpyDataset(X, y)
    dataloader = NumpyDataLoader(dataset, batch_size=batch_size, shuffle=False)

    # set torch model
    model = TorchMLP(struct[0], struct[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # set numpy model
    mlp = MultipleLayerPerceptron(struct=struct, n=n, model="classification")
    ce_loss = CrossEntropyLoss()

    # set numpy model weight as torch model weight in advance
    mlp.layers[0].w = model.fc.weight.T.detach().numpy().copy()
    mlp.layers[0].b = model.fc.bias.detach().numpy().copy()

    for _ in range(epoch):
        running_loss_np = 0.0
        running_loss_pt = 0.0

        for data in dataloader:
            X_train, y_train = data

            X_train = torch.tensor(torch.from_numpy(X_train.copy()))
            y_train = torch.tensor(
                torch.from_numpy(y_train.copy())
            )  # [2,0,4,1,2] label, which is not converted to one hot vector

            # torch implementation
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss_pt = criterion(y_pred, y_train)
            loss_pt.backward()
            optimizer.step()
            running_loss_pt += loss_pt.item()

            # numpy implementation
            X_train = X_train.detach().numpy()
            y_train = one_hot_vector(
                L, y_train.detach().numpy()
            )  # convert to one hot vector
            y_pred = mlp.forward(X_train)
            loss_np = ce_loss.forward(y_train, y_pred)
            dx_out = ce_loss.backward(y_train, y_pred)
            mlp.backward(dx_out)
            mlp.step(lr)
            running_loss_np += loss_np.item()

        running_loss_pt /= n_batch
        running_loss_np /= n_batch

        # check loss at every epoch
        np.testing.assert_almost_equal(running_loss_pt, running_loss_np)

    np.testing.assert_array_almost_equal(
        mlp.layers[0].w, model.fc.weight.T.detach().numpy()
    )
    np.testing.assert_array_almost_equal(
        mlp.layers[0].b, model.fc.bias.detach().numpy()
    )

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
from torch_model import TorchMLP, TorchCNN
from tools.utils import one_hot_vector
from data.data import NumpyDataset, NumpyDataLoader

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
    imgs_no_channel = torch.squeeze(imgs, 1)
    output_dim = 9
    y = np.eye(output_dim)[np.random.choice(output_dim, n)]
    kernel_dim = (3,3)
    padding = "same"
    pooling_size = 3
    epoch = 1
    lr = 0.1
    batch_size = 32
    n_batch = n // batch_size + 1
    res_loss = {}

    cnn = SingleCNN(input_dim=imgs_no_channel.shape,
                    output_dim=output_dim,
                    kernel_dim=kernel_dim,
                    padding=padding,
                    pooling_size=pooling_size)
    ce_loss = CrossEntropyLoss()

    # store weight
    kernel = torch.tensor(torch.from_numpy(cnn.cnn.kernel.copy()), dtype=torch.float64).unsqueeze(0).unsqueeze(0)
    cnn_bias = torch.tensor(torch.from_numpy(cnn.cnn.b.copy()), dtype=torch.float64)
    weight = torch.tensor(torch.from_numpy(cnn.fc.w.T.copy()), dtype=torch.float64)
    fc_bias = torch.tensor(torch.from_numpy(cnn.fc.b.copy()), dtype=torch.float64)
    kernel.requires_grad = True
    cnn_bias.requires_grad = True
    weight.requires_grad = True
    fc_bias.requires_grad = True

    # use numpy dataloader
    dataset = NumpyDataset(imgs, y)
    dataset_no_channel = NumpyDataset(imgs_no_channel, y)

    a = NumpyDataLoader(dataset_no_channel, batch_size=batch_size, shuffle=False)
    b = NumpyDataLoader(dataset, batch_size=batch_size, shuffle=False)
    for ((x,y),(x_,y_)) in zip(a,b):
        np.testing.assert_array_equal(y, y_)
        np.testing.assert_array_equal(x, np.squeeze(x_, axis=1))

    debug = {}
    for k in range(100):
        tmp = {
            "weight":{
                "fc_weight":{
                    "np":0,
                    "pt":0
                },
                "fc_bias": {
                    "np": 0,
                    "pt": 0
                },
                "conv_weight": {
                    "np": 0,
                    "pt": 0
                },
                "conv_bias": {
                    "np": 0,
                    "pt": 0
                }
            },
            "grad": {
                "fc_weight": {
                    "np": 0,
                    "pt": 0
                },
                "fc_bias": {
                    "np": 0,
                    "pt": 0
                },
                "conv_weight": {
                    "np": 0,
                    "pt": 0
                },
                "conv_bias": {
                    "np": 0,
                    "pt": 0
                }
            }
        }
        debug[k] = tmp

    # for i in range(epoch):
    #
    #     running_loss = 0.0
    #
    #     z = 0
    #     for data in NumpyDataLoader(dataset_no_channel, batch_size=batch_size, shuffle=False):
    #         X_train, y_train = data
    #
    #         y_pred_prob = cnn.forward(X_train) # not logits, probability prediction
    #         y_pred = y_pred_prob.argmax(axis=1)
    #         loss = ce_loss.forward(y_train, y_pred_prob)
    #         running_loss += loss.item()
    #
    #         dx_out = ce_loss.backward(y_train, y_pred_prob)
    #         cnn.backward(dx_out)
    #         cnn.step(lr)
    #
    #         debug[z]["weight"]["fc_weight"]["np"] = cnn.fc.w
    #         debug[z]["weight"]["fc_bias"]["np"] = cnn.fc.b
    #         debug[z]["grad"]["fc_weight"]["np"] = cnn.fc.dw
    #         debug[z]["grad"]["fc_bias"]["np"] = cnn.fc.db
    #
    #         debug[z]["weight"]["conv_weight"]["np"] = cnn.cnn.kernel
    #         debug[z]["weight"]["conv_bias"]["np"] = cnn.cnn.b
    #         debug[z]["grad"]["conv_weight"]["np"] = cnn.cnn.dk
    #         debug[z]["grad"]["conv_bias"]["np"] = cnn.cnn.db
    #
    #         z += 1
    #
    #
    #     print(running_loss)
    #
    #     # running_loss /= n_batch
    #     # res_loss["numpy"] = running_loss
    #
    #     # correct = (y_pred == y.argmax(axis=1)).sum()
    #
    # #     print(f"epoch: {i} / loss: {loss} / accuracy: {correct / 100 * 100}%")
    # # np_loss = loss

    ###### torch implementation ######
    # trainsets = TensorData(imgs, y)
    # trainloader = torch.utils.data.DataLoader(trainsets, batch_size=n)  # assuming batch gradient descent
    model = TorchCNN(h_in, w_in, output_dim, kernel_dim, pooling_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # model.conv.weight.data = kernel
    # model.conv.bias.data = cnn_bias
    # model.fc.weight.data = weight
    # model.fc.bias.data = fc_bias

    cnn.cnn.kernel = model.conv.weight.data.squeeze(0,1).detach().numpy().copy()
    cnn.cnn.b = model.conv.bias.data.detach().numpy().copy()
    cnn.fc.w = model.fc.weight.data.detach().numpy().copy().T
    cnn.fc.b = model.fc.bias.data.detach().numpy().copy()

    loss_ = []
    # n_batch = len(trainloader)

    for _ in range(epoch):

        running_loss = 0.0

        z = 0
        for data in NumpyDataLoader(dataset, batch_size=batch_size, shuffle=False):
        # for i, data in enumerate(trainloader, 0):
            X_train, y_train = data
            # X_train = torch.tensor(torch.from_numpy(X_train.copy()), dtype=torch.float32)
            y_train = torch.tensor(torch.from_numpy(y_train.copy()))

            optimizer.zero_grad()

            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            debug[z]["weight"]["fc_weight"]["pt"] = model.fc.weight.T
            debug[z]["weight"]["fc_bias"]["pt"] = model.fc.bias
            debug[z]["grad"]["fc_weight"]["pt"] = model.fc.weight.grad.T
            debug[z]["grad"]["fc_bias"]["pt"] = model.fc.bias.grad

            debug[z]["weight"]["conv_weight"]["pt"] = model.conv.weight
            debug[z]["weight"]["conv_bias"]["pt"] = model.conv.bias
            debug[z]["grad"]["conv_weight"]["pt"] = model.conv.weight.grad
            debug[z]["grad"]["conv_bias"]["pt"] = model.conv.bias.grad

            z += 1
            # break
            # if z == 2:
            #     break
        print(running_loss)
        # running_loss /= n_batch
        # res_loss["torch"] = running_loss

        loss_.append(running_loss / n_batch)

    # numpy implementation
    for i in range(epoch):

        running_loss = 0.0

        z = 0
        for data in NumpyDataLoader(dataset_no_channel, batch_size=batch_size, shuffle=False):
            X_train, y_train = data

            y_pred_prob = cnn.forward(X_train) # not logits, probability prediction
            y_pred = y_pred_prob.argmax(axis=1)
            loss = ce_loss.forward(y_train, y_pred_prob)
            running_loss += loss.item()

            dx_out = ce_loss.backward(y_train, y_pred_prob)
            cnn.backward(dx_out)
            cnn.step(lr)

            debug[z]["weight"]["fc_weight"]["np"] = cnn.fc.w
            debug[z]["weight"]["fc_bias"]["np"] = cnn.fc.b
            debug[z]["grad"]["fc_weight"]["np"] = cnn.fc.dw
            debug[z]["grad"]["fc_bias"]["np"] = cnn.fc.db

            debug[z]["weight"]["conv_weight"]["np"] = cnn.cnn.kernel
            debug[z]["weight"]["conv_bias"]["np"] = cnn.cnn.b
            debug[z]["grad"]["conv_weight"]["np"] = cnn.cnn.dk
            debug[z]["grad"]["conv_bias"]["np"] = cnn.cnn.db

            z += 1


        print(running_loss)

        # running_loss /= n_batch
        # res_loss["numpy"] = running_loss

        # correct = (y_pred == y.argmax(axis=1)).sum()

    #     print(f"epoch: {i} / loss: {loss} / accuracy: {correct / 100 * 100}%")
    # np_loss = loss


    print("hi")
    # # loss check
    # np.testing.assert_array_almost_equal(res_loss["torch"], res_loss["numpy"], decimal=3)
    # fc layer check
    np.testing.assert_array_almost_equal(model.fc.bias.detach().numpy(), cnn.fc.b, decimal=3)
    np.testing.assert_array_almost_equal(model.fc.weight.detach().numpy(), cnn.fc.w.T)
    # conv layer check
    np.testing.assert_array_almost_equal(model.conv.weight.squeeze((0, 1)).detach().numpy(), cnn.cnn.kernel)
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
        dx_out = mse_loss.backward(y, pred)
        loss_.append(loss)

        # backward
        mlp.backward(dx_out)

        # gradient descent
        mlp.step(0.001)

    ###### torch implementation ######
    trainsets = TensorData(X, y)
    trainloader = torch.utils.data.DataLoader(trainsets, batch_size=n) # assuming batch gradient descent
    model = TorchMLP(struct[0], struct[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.fc.weight.data = torch.tensor(torch.from_numpy(weight), dtype=torch.float32)
    model.fc.bias.data = torch.tensor(torch.from_numpy(bias), dtype=torch.float32)

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

    np.testing.assert_array_almost_equal(mlp.layers[0].dw, model.fc.weight.grad.numpy().T, decimal=6)
    np.testing.assert_array_almost_equal(mlp.layers[0].db, model.fc.bias.grad.numpy(), decimal=6)


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
    ce_loss = CrossEntropyLoss()

    # set weight, bias same as torch model
    mlp.layers[0].w = weight.T # (input_dim, output_dim)
    mlp.layers[0].b = bias

    loss_ = []
    for _ in range(epoch):
        # forward pass
        prob_pred = mlp.forward(X) # not logit, probability prediction

        # calculate loss
        loss = ce_loss.forward(y, prob_pred)
        dx_out = ce_loss.backward(y, prob_pred)
        loss_.append(loss)

        # backward
        mlp.backward(dx_out)

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
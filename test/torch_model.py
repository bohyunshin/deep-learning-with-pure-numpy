import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class TorchMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()  # 모델 연산 정의
        self.fc1 = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

if __name__ == '__main__':
    n = 64
    score = torch.randn(n,10)
    target = torch.randint(9, size=(64,))

    # print( torch.log(torch.sum(torch.exp(score))) - score[target])

    criterion = nn.CrossEntropyLoss()
    print(criterion(score, target))

    from tools.activations import Softmax
    from loss.classification import CrossEntropyLoss
    from tools.utils import one_hot_vector
    softmax = Softmax()
    ce = CrossEntropyLoss()
    prob = softmax.forward(np.array(score).reshape(-1, 10))
    one_hot = one_hot_vector(10, target)
    # one_hot[0][5] = 1
    # print(np.log(prob[0][5]))
    print(ce.forward(one_hot, prob))
    loss = criterion(score, torch.from_numpy(one_hot))
    print(loss)
    loss.backward()
    print("hi")
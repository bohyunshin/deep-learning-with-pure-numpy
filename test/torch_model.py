import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()  # 모델 연산 정의
        self.fc1 = nn.Linear(4, 1, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x
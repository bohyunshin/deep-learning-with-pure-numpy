import torch
from torch import nn
import torch.nn.functional as F


class TorchMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return x


class TorchCNN(nn.Module):
    def __init__(self, h_in, w_in, out_dim, kernel_size, pooling_size):
        super().__init__()
        self.pooling_size = pooling_size
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding="same")
        self.in_dim = (h_in // pooling_size) * (w_in // pooling_size)
        self.fc = nn.Linear(self.in_dim, out_dim)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=self.pooling_size, stride=self.pooling_size)
        x = F.relu(x)
        x = x.view(-1, self.in_dim)  # [batch_size, 1, h_in, w_in]
        x = self.fc(x)
        return x

if __name__ == "__main__":
    conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding="same")
    cnn = TorchCNN(4,4,3,3,2)
    imgs = torch.randn((1, 1, 4, 4))
    res = cnn(imgs)
    print("hi")
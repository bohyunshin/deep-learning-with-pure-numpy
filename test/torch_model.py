import torch
from torch import nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)


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
        conv_x = self.conv(x)
        pool_x = F.max_pool2d(conv_x, kernel_size=self.pooling_size, stride=self.pooling_size)
        relu_x = F.relu(pool_x)
        relu_x = relu_x.view(-1, self.in_dim)  # [batch_size, 1, h_in, w_in]
        fc_x = self.fc(relu_x)

        conv_x.retain_grad()
        pool_x.retain_grad()
        relu_x.retain_grad()
        fc_x.retain_grad()

        self.conv_x = conv_x
        self.pool_x = pool_x
        self.relu_x = relu_x
        self.fc_x = fc_x

        return fc_x

if __name__ == "__main__":
    conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding="same")
    cnn = TorchCNN(4,4,3,3,2)
    imgs = torch.randn((1, 1, 4, 4))
    res = cnn(imgs)
    print("hi")
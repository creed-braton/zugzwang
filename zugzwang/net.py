import torch
import torch.nn as nn
import torch.nn.functional as F
from .encode import POLICY_DIM


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class Net(nn.Module):
    def __init__(self, input_dim, res_blocks=6, channels=128):
        super().__init__()

        self.input_conv = nn.Conv2d(
            input_dim, channels, 3, padding=1, bias=False
        )
        self.input_bn = nn.BatchNorm2d(channels)

        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(res_blocks)]
        )

        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, POLICY_DIM)

        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value

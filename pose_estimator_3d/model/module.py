import torch
import torch.nn as nn


def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f'Activation "{name}" is invalid.')


class ResidualBlock(nn.Module):

    def __init__(self, hidden_size, activation='relu', dropout=0, residual=True, bias=False):
        super(ResidualBlock, self).__init__()

        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.activation = get_activation(activation)
        self.drop = nn.Dropout(dropout)
        self.residual = lambda x: x if residual else 0

    
    def forward(self, x):
        res = self.residual(x)
        x = self.drop(self.activation(self.bn1(self.fc1(x))))
        x = self.drop(self.activation(self.bn2(self.fc2(x))))
        return x + res


class DepthwiseSeparableConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=False, dilation=1, padding=0, stride=1):
        super(DepthwiseSeparableConv1d, self).__init__()

        self.depthwise_conv = nn.Conv1d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=kernel_size, groups=in_channels,
            bias=bias, stride=stride, padding=padding, dilation=dilation,)
        self.pointwise_conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, groups=1,
            bias=bias, stride=1, padding=0, dilation=1
        )


    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
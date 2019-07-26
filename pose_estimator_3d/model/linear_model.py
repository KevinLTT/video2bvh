from .module import ResidualBlock, get_activation

import torch
import torch.nn as nn


class LinearModel(nn.Module):

    def __init__(self, in_joint, in_channel, out_joint, out_channel, block_num, hidden_size,
                 activation='relu', dropout=0.25, bias=True, residual=True):
        super().__init__()

        self.in_joint = in_joint
        self.out_joint = out_joint
        self.out_channel = out_channel

        self.activation = get_activation(activation)
        self.drop = nn.Dropout(dropout)
        self.expand_fc = nn.Linear(in_joint*in_channel, hidden_size, bias=bias)
        self.expand_bn = nn.BatchNorm1d(hidden_size)
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_size, activation, dropout, residual, bias)
            for i in range(block_num)
        ])
        self.shrink_fc = nn.Linear(hidden_size, out_joint*out_channel, bias=bias)

    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        x = self.drop(self.activation(self.expand_bn(self.expand_fc(x))))
        x = self.blocks(x)
        x = self.shrink_fc(x)

        x = x.view(batch_size, self.out_joint, self.out_channel)
        return x
from .module import DepthwiseSeparableConv1d

import torch
import torch.nn as nn


class VideoPose(nn.Module):
    
    def __init__(self, in_joint, in_channel, out_joint, out_channel, filter_widths, hidden_size, dropout, dsc):
        super().__init__()

        self.train_model = None
        self.eval_model = TemporalModel(
            in_joint, in_channel, out_joint, out_channel, filter_widths, hidden_size, dropout, dsc
        )
        self.current_model = self.eval_model
    
    def forward(self, x):
        return self.current_model(x)


class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """
    
    def __init__(self, in_joint, in_channel, out_joint, out_channel, filter_widths, hidden_size, dropout, dsc):
        super().__init__()
        
        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'
        
        self.in_joint = in_joint
        self.out_joint = out_joint
        self.filter_widths = filter_widths
        self.out_channel = out_channel
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        self.pad = [ filter_widths[0] // 2 ]
        self.expand_bn = nn.BatchNorm1d(hidden_size, momentum=0.1)
        self.shrink = nn.Conv1d(hidden_size, out_joint * out_channel, 1)
        

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
        

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.in_joint
        
        batch_size, seq_len, joint, channel = x.shape
        x = x.view(batch_size, seq_len, -1)
        x = x.permute(0, 2, 1) # channel first

        x = self._forward_blocks(x)
            
        x = x.permute(0, 2, 1) # channel last
        x = x.view(batch_size, self.out_joint, self.out_channel)
        return x


class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """
    
    def __init__(self, in_joint, in_channel, out_joint, out_channel, filter_widths, hidden_size, dropout, dsc):
        super().__init__(in_joint, in_channel, out_joint, out_channel,
                         filter_widths, hidden_size, dropout, dsc)
        
        self.expand_conv = nn.Conv1d(in_joint*in_channel, hidden_size, filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        next_dilation = filter_widths[0]
        conv_class = DepthwiseSeparableConv1d if dsc else nn.Conv1d
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            layers_conv.append(conv_class(
                hidden_size, hidden_size, filter_widths[i], dilation=next_dilation, bias=False
            ))
            layers_bn.append(nn.BatchNorm1d(hidden_size, momentum=0.1))
            layers_conv.append(nn.Conv1d(hidden_size, hidden_size, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(hidden_size, momentum=0.1))
            
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            res = x[:, :, pad : x.shape[2] - pad]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)
        return x


class TemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.
    
    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """
    
    def __init__(self, in_joint, in_channel, out_joint, out_channel,
                 filter_widths, hidden_size, dropout, dsc):
        super().__init__(in_joint, in_channel, out_joint, out_channel,
                         filter_widths, hidden_size, dropout, dsc)
        
        self.expand_conv = nn.Conv1d(in_joint*in_channel, hidden_size, filter_widths[0],
                                     stride=filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        
        next_dilation = filter_widths[0]
        conv_class = DepthwiseSeparableConv1d if dsc else nn.Conv1d
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            layers_conv.append(conv_class(
                hidden_size, hidden_size, filter_widths[i], stride=filter_widths[i], bias=False
            ))
            layers_bn.append(nn.BatchNorm1d(hidden_size, momentum=0.1))
            layers_conv.append(nn.Conv1d(hidden_size, hidden_size, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(hidden_size, momentum=0.1))
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        
        for i in range(len(self.pad) - 1):
            res = x[:, :, self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.shrink(x)
        return x
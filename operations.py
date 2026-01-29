
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define all operators that DARTS can search over

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool1d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
    'max_pool_3x1': lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
    'max_pool_5x1': lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x1': lambda C, stride, affine: SepConv1D(C, C, 3, stride, 1, affine),
    'sep_conv_5x1': lambda C, stride, affine: SepConv1D(C, C, 5, stride, 2, affine),
    'sep_conv_1x3': lambda C, stride, affine: SepConv1D(C, C, 1, stride, 3, affine),
    'sep_conv_1x5': lambda C, stride, affine: SepConv1D(C, C, 1, stride, 5, affine),
    'dil_conv_3x1': lambda C, stride, affine: DilConv1D(C, C, 3, stride, 2, 2, affine),
    'dil_conv_5x1': lambda C, stride, affine: DilConv1D(C, C, 5, stride, 4, 2, affine),
    'conv_1x1': lambda C, stride, affine: ReLUConvBN1D(C, C, 1, stride, 0, affine),
    'conv_3x3': lambda C, stride, affine: ReLUConvBN1D(C, C, 3, stride, 1, affine),
}

# OPS = {
#     'none': lambda C, stride, affine: Zero(stride),
#     'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool1d(3, stride=stride, padding=1, count_include_pad=False),
#     'max_pool_3x3': lambda C, stride, affine: nn.MaxPool1d(3, stride=stride, padding=1),
#     'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
#     'sep_conv_1x3': lambda C, stride, affine: SepConv1D(C, C, 3, stride, 1, affine),
#     'sep_conv_1x5': lambda C, stride, affine: SepConv1D(C, C, 5, stride, 2, affine),
#     'dil_conv_1x3': lambda C, stride, affine: DilConv1D(C, C, 3, stride, 2, 2, affine),
#     'dil_conv_1x5': lambda C, stride, affine: DilConv1D(C, C, 5, stride, 4, 2, affine),
#     'conv_1x1': lambda C, stride, affine: ReLUConvBN1D(C, C, 1, stride, 0, affine),
#     'conv_3x3': lambda C, stride, affine: ReLUConvBN1D(C, C, 3, stride, 1, affine),
# }

class ReLUConvBN1D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class SepConv1D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_in, kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv1d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv1d(C_out, C_out, kernel_size, stride=1, padding=padding, groups=C_out, bias=False),
            nn.Conv1d(C_out, C_out, 1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class DilConv1D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv1d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        else:
            return x[:, :, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:])], dim=1)
        return self.bn(out)

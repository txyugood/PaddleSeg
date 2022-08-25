import math

from os.path import join as pjoin
from collections import OrderedDict

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return paddle.to_tensor(weights)


class StdConv2d(nn.Conv2D):
    def forward(self, x):
        if self._padding_mode != 'zeros':
            x = F.pad(x,
                      self._reversed_padding_repeated_twice,
                      mode=self._padding_mode,
                      data_format=self._data_format)

        w = self.weight
        v = paddle.var(w, axis=[1, 2, 3], keepdim=True, unbiased=False)
        m = paddle.mean(w, axis=[1, 2, 3], keepdim=True)
        w = (w - m) / paddle.sqrt(v + 1e-5)

        out = F.conv._conv_nd(
            x,
            w,
            bias=self.bias,
            stride=self._stride,
            padding=self._updated_padding,
            padding_algorithm=self._padding_algorithm,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format,
            channel_dim=self._channel_dim,
            op_type=self._op_type,
            use_cudnn=self._use_cudnn)
        return out


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(
        cin,
        cout,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias_attr=bias,
        groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(
        cin, cout, kernel_size=1, stride=stride, padding=0, bias_attr=bias)


class PreActBottleneck(nn.Layer):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, epsilon=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, epsilon=1e-6)
        self.conv2 = conv3x3(
            cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, epsilon=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU()

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(
            weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(
            weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(
            weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.set_value(conv1_weight)
        self.conv2.weight.set_value(conv2_weight)
        self.conv3.weight.set_value(conv3_weight)

        self.gn1.weight.set_value(gn1_weight.reshape([-1]))
        self.gn1.bias.set_value(gn1_bias.reshape([-1]))

        self.gn2.weight.set_value(gn2_weight.reshape([-1]))
        self.gn2.bias.set_value(gn2_bias.reshape([-1]))

        self.gn3.weight.set_value(gn3_weight.reshape([-1]))
        self.gn3.bias.set_value(gn3_bias.reshape([-1]))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(
                weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit,
                                                 "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit,
                                               "gn_proj/bias")])

            self.downsample.weight.set_value(proj_conv_weight)
            self.gn_proj.weight.set_value(proj_gn_weight.reshape([-1]))
            self.gn_proj.bias.set_value(proj_gn_bias.reshape([-1]))


class ResNetV2(nn.Layer):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(
            ('conv', StdConv2d(
                3, width, kernel_size=7, stride=2, bias_attr=False, padding=3)),
            ('gn', nn.GroupNorm(
                32, width, epsilon=1e-6)), ('relu', nn.ReLU()))

        self.body = nn.Sequential(
            ('block1', nn.Sequential(*([('unit1', PreActBottleneck(
                cin=width, cout=width * 4,
                cmid=width))] + [(f'unit{i:d}', PreActBottleneck(
                    cin=width * 4, cout=width * 4, cmid=width))
                                 for i in range(2, block_units[0] + 1)]))),
            ('block2', nn.Sequential(*([('unit1', PreActBottleneck(
                cin=width * 4, cout=width * 8, cmid=width * 2,
                stride=2))] + [(f'unit{i:d}', PreActBottleneck(
                    cin=width * 8, cout=width * 8, cmid=width * 2))
                               for i in range(2, block_units[1] + 1)]))),
            ('block3', nn.Sequential(*([('unit1', PreActBottleneck(
                cin=width * 8, cout=width * 16, cmid=width * 4,
                stride=2))] + [(f'unit{i:d}', PreActBottleneck(
                    cin=width * 16, cout=width * 16, cmid=width * 4))
                               for i in range(2, block_units[2] + 1)]))), )

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.shape
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2D(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.shape[2] == right_size:
                feat = x
            else:
                # pad = right_size - x.shape[2]
                # assert pad < 3 and pad > 0, "x {} should {}".format(x.shape, right_size)
                feat = paddle.zeros((b, x.shape[1], right_size, right_size))
                feat[:, :, 0:x.shape[2], 0:x.shape[3]] = x[:]
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]

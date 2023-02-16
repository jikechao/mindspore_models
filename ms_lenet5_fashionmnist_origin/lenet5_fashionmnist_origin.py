import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter


class Module1(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels):
        super(Module1, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(2, 2),
                                  stride=(1, 1),
                                  padding=(0, 1, 0, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Linear(nn.Cell):

    def __init__(self, matmul_0_w_shape, add_1_bias_shape):
        super(Linear, self).__init__()
        self.matmul_0_w = Parameter(Tensor(np.random.uniform(0, 1, matmul_0_w_shape).astype(np.float32)), name=None)
        self.add_1_bias = Parameter(Tensor(np.random.uniform(0, 1, add_1_bias_shape).astype(np.float32)), name=None)

    def construct(self, x):
        opt_matmul_0 = P.matmul(x, self.matmul_0_w)
        opt_add_1 = opt_matmul_0 + self.add_1_bias
        return opt_add_1


class MindSporeModel(nn.Cell):

    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.transpose_0 = P.Transpose()
        self.module1_0 = Module1(conv2d_0_in_channels=1, conv2d_0_out_channels=64)
        self.pad_maxpool2d_3 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.module1_1 = Module1(conv2d_0_in_channels=64, conv2d_0_out_channels=32)
        self.pad_maxpool2d_6 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_6 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.transpose_7 = P.Transpose()
        self.flatten_8 = nn.Flatten()
        self.linear_0 = Linear(matmul_0_w_shape=(1568, 120), add_1_bias_shape=(120, ))
        self.relu_11 = nn.ReLU()
        self.linear_1 = Linear(matmul_0_w_shape=(120, 84), add_1_bias_shape=(84, ))
        self.relu_14 = nn.ReLU()
        self.linear_2 = Linear(matmul_0_w_shape=(84, 10), add_1_bias_shape=(10, ))
        self.softmax_17 = nn.Softmax(axis=-1)

    def construct(self, conv2d_1_input):
        opt_transpose_0 = self.transpose_0(conv2d_1_input, (0, 3, 1, 2))
        module1_0_opt = self.module1_0(opt_transpose_0)
        opt_maxpool2d_3 = self.pad_maxpool2d_3(module1_0_opt)
        opt_maxpool2d_3 = self.maxpool2d_3(opt_maxpool2d_3)
        module1_1_opt = self.module1_1(opt_maxpool2d_3)
        opt_maxpool2d_6 = self.pad_maxpool2d_6(module1_1_opt)
        opt_maxpool2d_6 = self.maxpool2d_6(opt_maxpool2d_6)
        opt_transpose_7 = self.transpose_7(opt_maxpool2d_6, (0, 2, 3, 1))
        opt_flatten_8 = self.flatten_8(opt_transpose_7)
        linear_0_opt = self.linear_0(opt_flatten_8)
        opt_relu_11 = self.relu_11(linear_0_opt)
        linear_1_opt = self.linear_1(opt_relu_11)
        opt_relu_14 = self.relu_14(linear_1_opt)
        linear_2_opt = self.linear_2(opt_relu_14)
        opt_softmax_17 = self.softmax_17(linear_2_opt)
        return opt_softmax_17

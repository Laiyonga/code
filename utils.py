import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from struct import pack, unpack
from matplotlib.colors import ListedColormap


def yc_patch(A, l1, l2, o1, o2):
    n1, n2 = np.shape(A)
    tmp = np.mod(n1-l1, o1)
    if tmp != 0:
        A = np.concatenate([A, np.zeros((o1-tmp, n2))], axis=0)

    tmp = np.mod(n2-l2, o2)
    if tmp != 0:
        A = np.concatenate([A, np.zeros((A.shape[0], o2-tmp))], axis=-1)

    N1, N2 = np.shape(A)
    X = []
    for i1 in range(0, N1-l1+1, o1):
        for i2 in range(0, N2-l2+1, o2):
            tmp = np.reshape(A[i1:i1+l1, i2:i2+l2], (l1*l2, 1))
            X.append(tmp)
    X = np.array(X)
    return X[:, :, 0]


def yc_patch_inv(X1, n1, n2, l1, l2, o1, o2):
    tmp1 = np.mod(n1 - l1, o1)
    tmp2 = np.mod(n2 - l2, o2)
    if (tmp1 != 0) and (tmp2 != 0):
        A = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))
        mask = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))

    if (tmp1 != 0) and (tmp2 == 0):
        A = np.zeros((n1 + o1 - tmp1, n2))
        mask = np.zeros((n1 + o1 - tmp1, n2))

    if (tmp1 == 0) and (tmp2 != 0):
        A = np.zeros((n1, n2 + o2 - tmp2))
        mask = np.zeros((n1, n2 + o2 - tmp2))

    if (tmp1 == 0) and (tmp2 == 0):
        A = np.zeros((n1, n2))
        mask = np.zeros((n1, n2))

    N1, N2 = np.shape(A)
    ids = 0
    for i1 in range(0, N1 - l1 + 1, o1):
        for i2 in range(0, N2 - l2 + 1, o2):
            A[i1:i1 + l1, i2:i2 + l2] = A[i1:i1 + l1, i2:i2 + l2] + np.reshape(X1[:, ids], (l1, l2))
            mask[i1:i1 + l1, i2:i2 + l2] = mask[i1:i1 + l1, i2:i2 + l2] + np.ones((l1, l2))
            ids = ids + 1

    A = A / mask
    A = A[0:n1, 0:n2]
    return A


class BlockA(nn.Module):
    def __init__(self, in_chs, out_chs, dropout):
        super(BlockA, self).__init__()
        self.conv1 = nn.Conv2d(in_chs, out_chs, kernel_size=3, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_chs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_chs, mid_chs, dropout):
        super(DenseBlock, self).__init__()
        self.blocks = nn.ModuleList()

        for j in range(2):
            self.blocks.append(BlockA(in_chs, mid_chs, dropout))
            in_chs += mid_chs

    def forward(self, x):
        x1 = self.blocks[0](x)
        x2 = torch.cat([x, x1], dim=1)
        x3 = self.blocks[1](x2)
        skip_connection_list = [x1, x3]
        for j in range(2):
            if j == 0:
                out = skip_connection_list[j]
            else:
                out = torch.cat([out, skip_connection_list[j]], dim=1)
        return out


class TD(nn.Module):
    def __init__(self, in_chs, stride):
        super(TD, self).__init__()
        self.conv = nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(in_chs)
        self.stride = stride

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=self.stride)
        return x


class TU(nn.Module):
    def __init__(self, in_chs, stride, D1, dropout):
        super(TU, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_chs, D1, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batchnorm = nn.BatchNorm2d(D1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.deconv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class SK(nn.Module):
    def __init__(self, in_chs, m=2, r=8, L=32, kernel=4):
        super(SK, self).__init__()
        self.m = m
        self.r = r
        self.L = L
        self.kernel = kernel
        self.conv1 = nn.Conv2d(in_chs, kernel, kernel_size=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(kernel)
        self.conv2 = nn.Conv2d(kernel, kernel, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(kernel)
        self.conv3 = nn.Conv2d(kernel, kernel, kernel_size=5, stride=1, padding=2, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(kernel)
        self.global_avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.global_avg_pool2 = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(kernel, max(int(kernel / r), L))
        self.dense2 = nn.Linear(max(int(kernel / r), L), kernel * 2)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = F.relu(x)

        x1 = self.conv2(x)
        x1 = self.batchnorm2(x1)
        x1 = F.relu(x1)
        _x1 = self.global_avg_pool1(x1)

        x2 = self.conv3(x)
        x2 = self.batchnorm3(x2)
        x2 = F.relu(x2)
        _x2 = self.global_avg_pool2(x2)

        U = torch.add(_x1, _x2)
        U = torch.flatten(U, start_dim=1)
        z = self.dense1(U)
        z = F.relu(z)
        z = self.dense2(z)
        z = z.view(-1, self.kernel, 1, 1, self.m)
        scale = F.softmax(z, dim=-1)
        x = torch.stack([x1, x2], dim=-1)
        r = torch.mul(scale, x)
        r = torch.sum(r, dim=-1)
        return r


def cseis():
    seis = np.concatenate(
        (
            np.concatenate((0.5*np.ones([1, 40]), np.expand_dims(np.linspace(0.5, 1, 88), axis=1).transpose(),
                            np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])), axis=1).transpose(),
            np.concatenate((0.25*np.ones([1, 40]), np.expand_dims(np.linspace(0.25, 1, 88), axis=1).transpose(),
                            np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])), axis=1).transpose(),
            np.concatenate((np.zeros([1, 40]), np.expand_dims(np.linspace(0, 1, 88), axis=1).transpose(),
                            np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])), axis=1).transpose()
        ),
        axis=1)
    return ListedColormap(seis)


def ReadBinary(path):
    f = open(path, "rb")
    [startTime, endTime, xStart, xEnd] = unpack("4d", f.read(32))
    [nt, nx] = unpack("2i", f.read(8))
    [dt, dx] = unpack('2d', f.read(16))
    data = np.reshape(unpack(str(nt*nx)+"d", f.read(int(nt*nx)*8)), (nt, nx))
    f.close()
    # return [startTime, endTime, xStart, xEnd, nt, nx, dt, dx, data]
    return data
def ReadBinary1(path):
    f = open(path, "rb")
    [startTime, endTime, xStart, xEnd] = unpack("4d", f.read(32))
    [nt, nx] = unpack("2i", f.read(8))
    [dt, dx] = unpack('2d', f.read(16))
    data = np.reshape(unpack(str(nt*nx)+"d", f.read(int(nt*nx)*8)), (nt, nx))
    f.close()
    return [startTime, endTime, xStart, xEnd, nt, nx, dt, dx, data]
    #return data


if __name__ == '__main__':
    model_sk = SK(in_chs=16, kernel=8)
    input_data = torch.randn(1, 16, 64, 64)  # 替换为实际的输入数据
    print(model_sk(input_data))
















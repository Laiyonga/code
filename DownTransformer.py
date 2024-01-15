import time

import torch
import numbers
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class DownNet(nn.Module):
    def __init__(self, upscale_factor):
        super(DownNet, self).__init__()
        self.upscale_factor = upscale_factor
        self.init_feature = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.encoder1 = nn.Sequential(
            TransformerBlock(64, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            TransformerBlock(64, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            TransformerBlock(64, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0, bias=False)
        )
        self.encoder2 = nn.Sequential(
            TransformerBlock(128, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            TransformerBlock(128, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            TransformerBlock(128, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0, bias=False)
        )
        self.encoder3 = nn.Sequential(
            TransformerBlock(256, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            TransformerBlock(256, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            TransformerBlock(256, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            nn.Conv2d(256, 256, kernel_size=1, stride=2, padding=0, bias=False)
        )
        self.decoder1 = nn.Sequential(
            TransformerBlock(256, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            TransformerBlock(256, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            TransformerBlock(256, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.decoder2 = nn.Sequential(
            TransformerBlock(128, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            TransformerBlock(128, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            TransformerBlock(128, num_heads=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.decoder3 = nn.Sequential(
            TransformerBlock(64, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            TransformerBlock(64, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            TransformerBlock(64, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.proj = nn.Conv2d(64, 1, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.init_feature(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)

        x4 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = self.decoder1(x4)
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = self.decoder2(x5)
        x8 = self.proj(x6)
        out = x8

        return out


def cal_throughput(model, device):
    model.to(device)
    # 设置为评估模式
    model.eval()

    # 生成随机输入数据
    input_shape = (1, 1, 224, 224)  # 输入数据的形状 (batch_size, channels, height, width)
    input_data = torch.randn(*input_shape).to(device)

    # 进行预热阶段
    for _ in range(2):  # 预热阶段运行3次
        _ = model(input_data)

    print('Over!')
    # 使用CPU进行推理并计算吞吐量
    total_samples = input_shape[0]
    total_time = 0.0

    for _ in range(10):  # 进行10次运行取平均值
        start_time = time.time()
        outputs = model(input_data)
        end_time = time.time()
        total_time += end_time - start_time

    # 计算吞吐量
    throughput = total_samples * 50 / total_time

    # 打印吞吐量结果
    print(f"CPU Throughput: {throughput} samples/sec")


if __name__ == '__main__':
    import thop
    a = torch.randn(1, 1, 2990, 500)
    model = DownNet(upscale_factor=2)
    x_image = nn.Parameter(torch.randn(1, 1, 1496, 10), requires_grad=False)
    y = model(x_image)
    flops, params = thop.profile(model, inputs=(x_image,))
    flops = flops / 1e9
    params = params / 1e6
    print("FLOPs: {:.5f}G".format(flops))
    print("Parameters: {:.5f}M".format(params))
    # cal_throughput(model, device='cpu')





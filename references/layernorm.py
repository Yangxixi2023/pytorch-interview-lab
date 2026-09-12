"""实现 LayerNorm

输入：
x: 浮点 Tensor [..., D]
    只归一化最后一维，使用总体方差。
weight: 浮点 Tensor [D]
    缩放参数。
bias: 浮点 Tensor [D]
    偏移参数。
eps: float —
    加在方差内部的稳定项。

返回：
output: 浮点 Tensor 与 x 相同
    LayerNorm 的仿射输出；保留输入和参数梯度。
"""

import torch


def solve(x, weight, bias, eps=1e-5):
    feature_mean = x.mean(dim=-1, keepdim=True)
    centered = x - feature_mean
    # 使用总体方差，分母为 D，而不是 D-1。
    feature_variance = centered.square().mean(dim=-1, keepdim=True)
    normalized = centered * torch.rsqrt(feature_variance + eps)
    output = normalized * weight + bias
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(2, 3, 4, requires_grad=True),
        torch.randn(4, requires_grad=True),
        torch.randn(4, requires_grad=True),
    )
    print(solve(*args))

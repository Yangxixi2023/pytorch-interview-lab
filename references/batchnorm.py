"""实现 BatchNorm

输入：
x: 浮点 Tensor [N, C]
    N>1；沿 N 维统计。
weight: 浮点 Tensor [C]
    缩放。
bias: 浮点 Tensor [C]
    偏移。
running_mean: 浮点 Tensor [C]
    历史均值。
running_var: 浮点 Tensor [C]
    历史方差。
training: bool —
    是否使用 batch 统计。
momentum: float —
    新统计在 running 更新中的权重。
eps: float —
    方差稳定项。

返回：
output: 浮点 Tensor [N, C]
    训练归一化使用总体方差，保留梯度。
new_running_mean: 浮点 Tensor [C]
    更新后的均值；训练统计 detach，不修改输入。
new_running_var: 浮点 Tensor [C]
    更新使用无偏方差；训练统计 detach，不修改输入。
多个返回值按上面顺序组成元组。
"""

import torch


def solve(
    x,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum=0.1,
    eps=1e-5,
):
    if training:
        mean = x.mean(dim=0)
        variance = x.var(dim=0, unbiased=False)

        # 输出用总体方差；running_var 的更新使用无偏方差。
        detached_mean = mean.detach()
        unbiased_variance = x.detach().var(dim=0, unbiased=True)
        new_running_mean = (
            1 - momentum
        ) * running_mean + momentum * detached_mean
        new_running_var = (
            1 - momentum
        ) * running_var + momentum * unbiased_variance
    else:
        mean = running_mean
        variance = running_var
        new_running_mean = running_mean
        new_running_var = running_var

    # 推理时直接使用 running 统计，不再依赖当前 batch。
    normalized = (x - mean) * torch.rsqrt(variance + eps)
    output = normalized * weight + bias
    return output, new_running_mean, new_running_var


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(5, 3, requires_grad=True),
        torch.ones(3),
        torch.zeros(3),
        torch.zeros(3),
        torch.ones(3),
        True,
    )
    print(solve(*args))

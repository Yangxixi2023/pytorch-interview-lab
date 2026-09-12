"""实现 BatchNorm

输入：
x: 浮点 Tensor [N, C] — N>1；沿 N 维统计。
weight: 浮点 Tensor [C] — 缩放。
bias: 浮点 Tensor [C] — 偏移。
running_mean: 浮点 Tensor [C] — 历史均值。
running_var: 浮点 Tensor [C] — 历史方差。
training: bool — — 是否使用 batch 统计。
momentum: float — — 新统计在 running 更新中的权重。
eps: float — — 方差稳定项。

返回（多项按元组顺序）：
output: 浮点 Tensor [N, C] — 训练归一化使用总体方差，保留梯度。
new_running_mean: 浮点 Tensor [C] — 更新后的均值；训练统计 detach，不修改输入。
new_running_var: 浮点 Tensor [C] — 更新使用无偏方差；训练统计 detach，不修改输入。
"""

import torch

def solve(x, weight, bias, running_mean, running_var, training, momentum=0.1, eps=1e-05):
    if training:
        mean = x.mean(0)
        var = x.var(0, unbiased=False)
        rm = (1 - momentum) * running_mean + momentum * mean.detach()
        rv = (1 - momentum) * running_var + momentum * x.detach().var(0, unbiased=True)
    else:
        mean = running_mean
        var = running_var
        rm = running_mean
        rv = running_var
    return ((x - mean) * torch.rsqrt(var + eps) * weight + bias, rm, rv)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(5, 3, requires_grad=True), torch.ones(3), torch.zeros(3), torch.zeros(3), torch.ones(3), True)
    print(solve(*args))

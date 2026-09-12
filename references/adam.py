"""Adam 优化器

输入：
param: 浮点 Tensor 与 param 相同 — 当前参数。
grad: 浮点 Tensor 与 param 相同 — 当前梯度。
m: 浮点 Tensor 与 param 相同 — 上一时刻一阶动量。
v: 浮点 Tensor 与 param 相同 — 上一时刻二阶动量。
step: int — — 本次更新步数，从 1 开始。
lr: float — — 学习率。
beta1: float — — 一阶动量衰减系数。
beta2: float — — 二阶动量衰减系数。
eps: float — — 稳定项，放在 sqrt 外。

返回（多项按元组顺序）：
new_param: 浮点 Tensor 与 param 相同 — 更新后的参数，不修改 param。
new_m: 浮点 Tensor 与 m 相同 — 更新后的一阶动量。
new_v: 浮点 Tensor 与 v 相同 — 更新后的二阶动量。
"""

import torch

def solve(param, grad, m, v, step, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-08):
    m1 = beta1 * m + (1 - beta1) * grad
    v1 = beta2 * v + (1 - beta2) * grad.square()
    p = param - lr * (m1 / (1 - beta1 ** step)) / ((v1 / (1 - beta2 ** step)).sqrt() + eps)
    return (p, m1, v1)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.ones(3), torch.tensor([1.0, -2.0, 0.0]), torch.zeros(3), torch.zeros(3), 1)
    print(solve(*args))

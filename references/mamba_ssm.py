"""Mamba SSM 步骤

输入：
u: 浮点 Tensor [B, T, D] — 序列输入。
delta: 浮点 Tensor [B, T, D] — 每步离散化步长。
a: 浮点 Tensor [D, N] — 状态转移参数。
b: 浮点 Tensor [B, T, N] — 选择性输入参数。
c: 浮点 Tensor [B, T, N] — 选择性读出参数。
d: 浮点 Tensor [D] — 输入跳连系数。
initial: 浮点 Tensor [B, D, N] — 初始状态，可以非零。

返回（多项按元组顺序）：
y: 浮点 Tensor [B, T, D] — 每步状态读出加输入跳连。
final_h: 浮点 Tensor [B, D, N] — 最后一个时间步之后的状态。
"""

import torch

def solve(u, delta, a, b, c, d, initial):
    h = initial
    ys = []
    for t in range(u.shape[1]):
        dt = delta[:, t, :, None]
        h = (dt * a).exp() * h + dt * b[:, t, None, :] * u[:, t, :, None]
        ys.append((h * c[:, t, None, :]).sum(-1) + d * u[:, t])
    return (torch.stack(ys, 1), h)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 4, 3, requires_grad=True), torch.rand(2, 4, 3), -torch.rand(3, 2), torch.randn(2, 4, 2), torch.randn(2, 4, 2), torch.randn(3), torch.zeros(2, 3, 2))
    print(solve(*args))

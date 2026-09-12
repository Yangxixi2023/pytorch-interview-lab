"""梯度范数裁剪

输入：
grads: list[Tensor] 各元素形状可不同 — 所有需要一起计算全局 L2 范数的梯度。
max_norm: float — — 允许的最大范数。
eps: float — — 加在分母中的稳定项。

返回（多项按元组顺序）：
clipped_grads: list[Tensor] 与 grads 一一对应 — 裁剪后的新梯度列表，不原地修改输入。
total_norm: 浮点 Tensor [] — 裁剪之前的全局 L2 范数。
"""

import torch

def solve(grads, max_norm, eps=1e-06):
    norm = torch.sqrt(sum((g.square().sum() for g in grads)))
    scale = (max_norm / (norm + eps)).clamp(max=1)
    return ([g * scale for g in grads], norm)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = ([torch.tensor([3.0, 4.0]), torch.tensor([12.0])], 5.0)
    print(solve(*args))

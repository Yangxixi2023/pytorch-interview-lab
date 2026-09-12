"""梯度范数裁剪

输入：
grads: list[Tensor] 各元素形状可不同
    所有需要一起计算全局 L2 范数的梯度。
max_norm: float —
    允许的最大范数。
eps: float —
    加在分母中的稳定项。

返回：
clipped_grads: list[Tensor] 与 grads 一一对应
    裁剪后的新梯度列表，不原地修改输入。
total_norm: 浮点 Tensor []
    裁剪之前的全局 L2 范数。
多个返回值按上面顺序组成元组。
"""

import torch


def solve(grads, max_norm, eps=1e-6):
    # 所有参数的梯度拼在一起考虑，而不是每个张量独立裁剪。
    squared_norm = sum(gradient.square().sum() for gradient in grads)
    total_norm = torch.sqrt(squared_norm)
    scale = torch.clamp(max_norm / (total_norm + eps), max=1.0)

    clipped_gradients = []
    for gradient in grads:
        clipped_gradients.append(gradient * scale)
    return clipped_gradients, total_norm


if __name__ == "__main__":
    torch.manual_seed(17)
    args = ([torch.tensor([3.0, 4.0]), torch.tensor([12.0])], 5.0)
    print(solve(*args))

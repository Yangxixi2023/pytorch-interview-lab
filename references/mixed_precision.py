"""混合精度训练步骤

输入：
param: 浮点 Tensor 任意形状 — float32 参数，不原地修改。
scaled_grad: 浮点 Tensor 与 param 相同 — 经过 loss scaling 的梯度，可能包含 NaN/Inf。
loss_scale: float — — 正的 loss scaling 系数。
lr: float — — 学习率。

返回（多项按元组顺序）：
new_param: 浮点 Tensor 与 param 相同 — 有限梯度时更新，否则复制原参数。
updated: bool — — 更新成功返回 True，出现非有限梯度并跳过返回 False。
"""

import torch

def solve(param, scaled_grad, loss_scale, lr):
    g = scaled_grad / loss_scale
    if not torch.isfinite(g).all():
        return (param.clone(), False)
    return (param - lr * g, True)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.ones(3), torch.tensor([128.0, -256.0, 0.0]), 128.0, 0.1)
    print(solve(*args))

"""Focal Loss

输入：
logits: 浮点 Tensor [N, C] — 未经 softmax 的分类分数。
targets: int64 Tensor [N] — 每个样本的正确类别，取值 0..C-1。
gamma: float — — Focal 调制指数；gamma=0 退化为交叉熵。

返回（多项按元组顺序）：
loss: 浮点 Tensor [] — 零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch

def solve(logits, targets, gamma=2.0):
    lp = logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
    return (-(1 - lp.exp()) ** gamma * lp).mean()

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(6, 4, requires_grad=True), torch.tensor([0, 1, 2, 3, 0, 1]))
    print(solve(*args))

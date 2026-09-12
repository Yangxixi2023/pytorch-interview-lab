"""图自编码器 GAE

输入：
z: 浮点 Tensor [N, D] — 给定的节点编码，不需要实现编码器。
adjacency: 浮点 Tensor [N, N] — 0/1 重构目标，包含对角线。

返回（多项按元组顺序）：
loss: 浮点 Tensor [] — 零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch
import torch.nn.functional as F

def solve(z, adjacency):
    logits = z @ z.T
    return (F.softplus(logits) - adjacency * logits).mean()

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(4, 3, requires_grad=True), torch.eye(4))
    print(solve(*args))

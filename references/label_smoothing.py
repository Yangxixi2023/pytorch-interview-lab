"""标签平滑损失

输入：
logits: 浮点 Tensor [N, C] — 未经 softmax 的分类分数。
targets: int64 Tensor [N] — 每个样本的正确类别，取值 0..C-1。
smoothing: float — — 平滑比例，标签分布为 (1-smoothing)*one_hot+smoothing/C。

返回（多项按元组顺序）：
loss: 浮点 Tensor [] — 零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch

def solve(logits, targets, smoothing=0.1):
    lp = logits.log_softmax(-1)
    return -(1 - smoothing) * lp.gather(1, targets[:, None]).mean() - smoothing * lp.mean()

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(4, 5, requires_grad=True), torch.tensor([0, 2, 3, 1]), 0.2)
    print(solve(*args))

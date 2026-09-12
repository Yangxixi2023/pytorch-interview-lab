"""交叉熵损失

输入：
logits: 浮点 Tensor [N, C] — 未经 softmax 的分类分数。
targets: int64 Tensor [N] — 每个样本的正确类别，取值 0..C-1。

返回（多项按元组顺序）：
loss: 浮点 Tensor [] — 零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch

def solve(logits, targets):
    return (torch.logsumexp(logits, dim=-1) - logits.gather(1, targets[:, None]).squeeze(1)).mean()

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(5, 4, requires_grad=True), torch.tensor([0, 1, 3, 2, 1]))
    print(solve(*args))

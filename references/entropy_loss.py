"""Entropy Loss 策略熵

输入：
logits: 浮点 Tensor [B, T, V] — 未归一化的词表分数。
mask: bool Tensor [B, T] — 至少一个 True；仅在有效位置平均。

返回（多项按元组顺序）：
entropy: 浮点 Tensor [] — 正号平均熵，零维可导张量；不是负熵正则 loss。
"""

import torch

def solve(logits, mask):
    lp = logits.log_softmax(-1)
    entropy = -(lp.exp() * lp).sum(-1)
    return entropy[mask].mean()

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.zeros(2, 3, 4, requires_grad=True), torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool))
    print(solve(*args))

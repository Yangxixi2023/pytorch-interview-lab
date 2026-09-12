"""Entropy Loss 策略熵

输入：
logits: 浮点 Tensor [B, T, V]
    未归一化的词表分数。
mask: bool Tensor [B, T]
    至少一个 True；仅在有效位置平均。

返回：
entropy: 浮点 Tensor []
    正号平均熵，零维可导张量；不是负熵正则 loss。
"""

import torch


def solve(logits, mask):
    log_probabilities = logits.log_softmax(dim=-1)  # [B,T,V]
    probabilities = log_probabilities.exp()
    token_entropy = -(probabilities * log_probabilities).sum(dim=-1)

    # 返回正号熵；训练时如需鼓励探索，应在总 loss 中减去它。
    mean_entropy = token_entropy[mask].mean()
    return mean_entropy


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.zeros(2, 3, 4, requires_grad=True),
        torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool),
    )
    print(solve(*args))

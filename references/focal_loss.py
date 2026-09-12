"""Focal Loss

输入：
logits: 浮点 Tensor [N, C]
    未经 softmax 的分类分数。
targets: int64 Tensor [N]
    每个样本的正确类别，取值 0..C-1。
gamma: float —
    Focal 调制指数；gamma=0 退化为交叉熵。

返回：
loss: 浮点 Tensor []
    零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch


def solve(logits, targets, gamma=2.0):
    log_probabilities = logits.log_softmax(dim=-1)
    # 只提取每个样本正确类别的概率，用它衡量样本难度。
    target_log_probabilities = log_probabilities.gather(1, targets[:, None])
    target_log_probabilities = target_log_probabilities.squeeze(-1)
    target_probabilities = target_log_probabilities.exp()

    # 正确类别概率高的样本，被 (1-pt)^gamma 降低权重。
    modulation = (1.0 - target_probabilities).pow(gamma)
    sample_losses = -modulation * target_log_probabilities
    return sample_losses.mean()


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(6, 4, requires_grad=True),
        torch.tensor([0, 1, 2, 3, 0, 1]),
    )
    print(solve(*args))

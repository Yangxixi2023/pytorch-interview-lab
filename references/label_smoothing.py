"""标签平滑损失

输入：
logits: 浮点 Tensor [N, C]
    未经 softmax 的分类分数。
targets: int64 Tensor [N]
    每个样本的正确类别，取值 0..C-1。
smoothing: float —
    平滑比例，标签分布为 (1-smoothing)*one_hot+smoothing/C。

返回：
loss: 浮点 Tensor []
    零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch


def solve(logits, targets, smoothing=0.1):
    log_probabilities = logits.log_softmax(dim=-1)
    target_indices = targets.unsqueeze(-1)
    target_log_probabilities = log_probabilities.gather(1, target_indices)

    # 平滑目标是 (1-smoothing)*one_hot + smoothing/C。
    hard_label_loss = -target_log_probabilities.mean()
    uniform_label_loss = -log_probabilities.mean()
    loss = (
        1 - smoothing
    ) * hard_label_loss + smoothing * uniform_label_loss
    return loss


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(4, 5, requires_grad=True),
        torch.tensor([0, 2, 3, 1]),
        0.2,
    )
    print(solve(*args))

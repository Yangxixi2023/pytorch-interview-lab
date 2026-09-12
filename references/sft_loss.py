"""SFT Loss 因果语言模型

输入：
logits: 浮点 Tensor [B, T, V]
    t 位置预测 t+1 标签，T>=2。
labels: int64 Tensor [B, T]
    目标 token ID，或 ignore_index；shift 后至少一个有效位置。
ignore_index: int —
    屏蔽 prompt/padding 的特殊标签值。

返回：
loss: 浮点 Tensor []
    零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch


def solve(logits, labels, ignore_index=-100):
    # 位置 t 的 logits 预测位置 t+1 的标签。
    prediction_logits = logits[:, :-1, :]  # [B,T-1,V]
    next_token_labels = labels[:, 1:]  # [B,T-1]
    valid_positions = next_token_labels != ignore_index

    # 先筛掉 prompt/padding，再索引正确类别，避免拿 -100 去 gather。
    valid_logits = prediction_logits[valid_positions]  # [Nvalid,V]
    valid_labels = next_token_labels[valid_positions]  # [Nvalid]
    log_probabilities = valid_logits.log_softmax(dim=-1)
    target_log_probabilities = log_probabilities.gather(
        1, valid_labels[:, None]
    )
    loss = -target_log_probabilities.mean()
    return loss


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(2, 4, 5, requires_grad=True),
        torch.tensor([[-100, -100, 2, 1], [-100, 3, 0, -100]]),
    )
    print(solve(*args))

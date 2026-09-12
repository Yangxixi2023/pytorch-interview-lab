"""多 Token 预测

输入：
logits: 浮点 Tensor [K, B, T, V]
    第 k 个 head 在 t 位置预测 t+k+1；k 从0开始。
tokens: int64 Tensor [B, T]
    目标 token ID；保证 T>K。

返回：
loss: 浮点 Tensor []
    零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch
import torch.nn.functional as F


def solve(logits, tokens):
    head_count, batch_size, sequence_length, vocabulary_size = logits.shape
    total_loss = logits.new_zeros(())
    total_targets = 0

    for head_index in range(head_count):
        prediction_offset = head_index + 1
        target_tokens = tokens[:, prediction_offset:]
        head_logits = logits[head_index, :, :-prediction_offset, :]

        # 每个 head 剩余的有效位置数不同，不能直接平均各 head loss。
        flattened_logits = head_logits.reshape(-1, vocabulary_size)
        flattened_targets = target_tokens.reshape(-1)
        head_loss = F.cross_entropy(
            flattened_logits, flattened_targets, reduction="sum"
        )
        total_loss = total_loss + head_loss
        total_targets += flattened_targets.numel()

    return total_loss / total_targets


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(3, 2, 5, 7, requires_grad=True),
        torch.randint(7, (2, 5)),
    )
    print(solve(*args))

"""滑动窗口注意力

输入：
q: 浮点 Tensor [B, H, T, D]
    已拆头的 query。
k: 浮点 Tensor [B, H, T, D]
    已拆头的 key。
v: 浮点 Tensor [B, H, T, D]
    已拆头的 value。
window: int —
    可见窗口长度，包含当前位置；为正整数。

返回：
output: 浮点 Tensor [B, H, T, D]
    因果注意力输出；不允许读取未来位置，保留梯度。
"""

import math
import torch


def solve(q, k, v, window=3):
    sequence_length = q.shape[-2]
    positions = torch.arange(sequence_length, device=q.device)
    query_positions = positions[:, None]
    key_positions = positions[None, :]

    is_future = key_positions > query_positions
    is_too_old = key_positions < query_positions - window + 1
    # 同时屏蔽未来和过旧的位置；window 包含当前位置。
    blocked_positions = is_future | is_too_old

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(blocked_positions, float("-inf"))
    attention_weights = torch.softmax(scores, dim=-1)
    return attention_weights @ v


if __name__ == "__main__":
    torch.manual_seed(17)
    args = tuple(
        (torch.randn(2, 2, 5, 4, requires_grad=True) for _ in range(3))
    )
    print(solve(*args))

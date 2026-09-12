"""ALiBi 注意力

输入：
q: 浮点 Tensor [B, H, T, D]
    已拆头的 query。
k: 浮点 Tensor [B, H, T, D]
    已拆头的 key。
v: 浮点 Tensor [B, H, T, D]
    已拆头的 value。
slopes: 浮点 Tensor [H]
    每个 head 的位置惩罚斜率。

返回：
output: 浮点 Tensor [B, H, T, D]
    因果注意力输出；不允许读取未来位置，保留梯度。
"""

import math
import torch


def solve(q, k, v, slopes):
    sequence_length = q.shape[-2]
    positions = torch.arange(sequence_length, device=q.device)
    query_positions = positions[:, None]
    key_positions = positions[None, :]
    relative_distance = query_positions - key_positions  # [T,T]

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    # 每个 head 使用自己的斜率，惩罚距离较远的历史位置。
    position_penalty = slopes[None, :, None, None] * relative_distance
    scores = scores - position_penalty
    scores = scores.masked_fill(
        key_positions > query_positions, float("-inf")
    )
    attention_weights = torch.softmax(scores, dim=-1)
    return attention_weights @ v


if __name__ == "__main__":
    torch.manual_seed(17)
    args = tuple(
        (torch.randn(2, 2, 5, 4, requires_grad=True) for _ in range(3))
    ) + (torch.tensor([0.1, 0.3]),)
    print(solve(*args))

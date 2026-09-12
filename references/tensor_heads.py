"""张量变换与多头重排

输入：
x: 浮点 Tensor [B, T, D]
    可能不是连续张量，D 可被 heads 整除。
heads: int —
    head 数 H，正整数。

返回：
split_heads: 浮点 Tensor [B, H, T, D/H]
    拆头并交换时间与头维度，保留梯度。
merged: 浮点 Tensor [B, T, D]
    合并还原的张量，值与输入一致。
多个返回值按上面顺序组成元组。
"""

import torch


def solve(x, heads):
    batch_size, sequence_length, hidden_dim = x.shape
    head_dim = hidden_dim // heads

    split = x.reshape(batch_size, sequence_length, heads, head_dim)
    split_heads = split.transpose(1, 2)  # [B,H,T,Dh]

    # 合并前先交换回时间维；否则 reshape 会混淆 token 与 head。
    time_major_heads = split_heads.transpose(1, 2)
    merged = time_major_heads.reshape(
        batch_size, sequence_length, hidden_dim
    )
    return split_heads, merged


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.arange(48.0).reshape(2, 2, 12).requires_grad_(), 3)
    print(solve(*args))

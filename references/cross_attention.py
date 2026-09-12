"""多头交叉注意力

输入：
q: 浮点 Tensor [B, H, T, D]
    已投影的 query。
k: 浮点 Tensor [B, H, S, D]
    已投影的 key，S 可以不等于 T。
v: 浮点 Tensor [B, H, S, D]
    已投影的 value。

返回：
output: 浮点 Tensor [B, H, T, D]
    非因果注意力；不合并 head 维。
"""

import math
import torch


def solve(q, k, v):
    # query 与 key 的序列长度可以不同，头数与 head_dim 必须兼容。
    head_dim = q.shape[-1]
    scores = q @ k.transpose(-2, -1)  # [B,H,T,S]
    scores = scores / math.sqrt(head_dim)
    attention_weights = torch.softmax(scores, dim=-1)
    output = attention_weights @ v  # [B,H,T,D]
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(2, 3, 4, 5, requires_grad=True),
        torch.randn(2, 3, 7, 5, requires_grad=True),
        torch.randn(2, 3, 7, 5, requires_grad=True),
    )
    print(solve(*args))

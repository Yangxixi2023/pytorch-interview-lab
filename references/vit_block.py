"""ViT Transformer Block

输入：
x: 浮点 Tensor [B, T, D]
    block 输入。
wqkv: 浮点 Tensor [D, 3D]
    联合 QKV 投影，按最后一维分成 q/k/v。
wo: 浮点 Tensor [D, D]
    注意力输出投影。
w1: 浮点 Tensor [D, F]
    MLP 上投影。
w2: 浮点 Tensor [F, D]
    MLP 下投影。
heads: int —
    head 数，必须整除 D。
eps: float —
    两处无仿射 LayerNorm 的稳定项。

返回：
output: 浮点 Tensor [B, T, D]
    pre-LN 双残差 block 输出；GPT-2 因果，ViT 双向。
"""

import math
import torch
import torch.nn.functional as F


def solve(x, wqkv, wo, w1, w2, heads, eps=1e-5):
    batch_size, sequence_length, hidden_dim = x.shape
    head_dim = hidden_dim // heads

    # Pre-LN：注意力子层之前归一化，残差保留原始 x。
    normalized_input = F.layer_norm(x, (hidden_dim,), eps=eps)
    query, key, value = (normalized_input @ wqkv).chunk(3, dim=-1)
    query = query.reshape(
        batch_size, sequence_length, heads, head_dim
    ).transpose(1, 2)
    key = key.reshape(
        batch_size, sequence_length, heads, head_dim
    ).transpose(1, 2)
    value = value.reshape(
        batch_size, sequence_length, heads, head_dim
    ).transpose(1, 2)
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)

    attention_weights = scores.softmax(dim=-1)
    head_outputs = attention_weights @ value
    merged = head_outputs.transpose(1, 2)
    merged = merged.reshape(batch_size, sequence_length, hidden_dim)
    attention_residual = x + merged @ wo

    # 第二个 Pre-LN 子层：精确 GELU 的 FFN，再做一次残差相加。
    normalized_residual = F.layer_norm(
        attention_residual, (hidden_dim,), eps=eps
    )
    hidden_features = F.gelu(normalized_residual @ w1)
    output = attention_residual + hidden_features @ w2
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(2, 3, 4, requires_grad=True),
        torch.randn(4, 12, requires_grad=True) * 0.2,
        torch.randn(4, 4, requires_grad=True) * 0.2,
        torch.randn(4, 6, requires_grad=True) * 0.2,
        torch.randn(6, 4, requires_grad=True) * 0.2,
        2,
    )
    print(solve(*args))

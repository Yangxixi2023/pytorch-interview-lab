"""线性自注意力

输入：
q: 浮点 Tensor [B, T, D]
    应用 ELU+1 特征映射的 query。
k: 浮点 Tensor [B, T, D]
    应用 ELU+1 特征映射的 key。
v: 浮点 Tensor [B, T, D]
    value 不做特征映射。
eps: float —
    归一化分母的稳定项。

返回：
output: 浮点 Tensor [B, T, D]
    非因果线性注意力，不构造 T×T 矩阵。
"""

import torch
import torch.nn.functional as F


def solve(q, k, v, eps=1e-6):
    query_features = F.elu(q) + 1.0
    key_features = F.elu(k) + 1.0

    # 先计算 [D,D] 的 K^T V，避免构造 [T,T] 的注意力矩阵。
    key_value_summary = key_features.transpose(-2, -1) @ v
    weighted_values = query_features @ key_value_summary
    key_sum = key_features.sum(dim=-2, keepdim=True)
    normalizer = (query_features * key_sum).sum(dim=-1, keepdim=True)
    output = weighted_values / (normalizer + eps)
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = tuple(
        (torch.randn(2, 5, 3, requires_grad=True) for _ in range(3))
    )
    print(solve(*args))

"""线性自注意力

输入：
q: 浮点 Tensor [B, T, D] — 应用 ELU+1 特征映射的 query。
k: 浮点 Tensor [B, T, D] — 应用 ELU+1 特征映射的 key。
v: 浮点 Tensor [B, T, D] — value 不做特征映射。
eps: float — — 归一化分母的稳定项。

返回（多项按元组顺序）：
output: 浮点 Tensor [B, T, D] — 非因果线性注意力，不构造 T×T 矩阵。
"""

import torch
import torch.nn.functional as F

def solve(q, k, v, eps=1e-06):
    q = F.elu(q) + 1
    k = F.elu(k) + 1
    kv = k.transpose(-2, -1) @ v
    return q @ kv / ((q * k.sum(-2, keepdim=True)).sum(-1, keepdim=True) + eps)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = tuple((torch.randn(2, 5, 3, requires_grad=True) for _ in range(3)))
    print(solve(*args))

"""Flash Attention 分块

输入：
q: 浮点 Tensor [Q, D] — 单头 query，无 batch 维。
k: 浮点 Tensor [K, D] — 按序列维分块处理的 key。
v: 浮点 Tensor [K, V] — 按同样块划分的 value。
block_size: int — — 正的 KV 分块大小，K 不一定可整除。

返回（多项按元组顺序）：
output: 浮点 Tensor [Q, V] — 精确非因果注意力，不构造完整 Q×K 分数矩阵。
"""

import math
import torch

def solve(q, k, v, block_size=2):
    m = q.new_full((len(q), 1), float('-inf'))
    l = q.new_zeros(len(q), 1)
    acc = q.new_zeros(len(q), v.shape[-1])
    for start in range(0, len(k), block_size):
        s = q @ k[start:start + block_size].T / math.sqrt(q.shape[-1])
        new_m = torch.maximum(m, s.amax(-1, keepdim=True))
        scale = (m - new_m).exp()
        p = (s - new_m).exp()
        acc = acc * scale + p @ v[start:start + block_size]
        l = l * scale + p.sum(-1, keepdim=True)
        m = new_m
    return acc / l

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(4, 3, requires_grad=True), torch.randn(7, 3, requires_grad=True), torch.randn(7, 5, requires_grad=True), 3)
    print(solve(*args))

"""分组查询注意力 GQA

输入：
q: 浮点 Tensor [B, Hq, T, D] — Hq 必须可被 Hkv 整除。
k: 浮点 Tensor [B, Hkv, S, D] — 连续 Hq/Hkv 个 query head 共用一个 KV head。
v: 浮点 Tensor [B, Hkv, S, D] — 与 key 的头数和序列长度一致。

返回（多项按元组顺序）：
output: 浮点 Tensor [B, Hq, T, D] — 非因果 GQA 输出，不合并 head 维。
"""

import math
import torch

def solve(q, k, v):
    n = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(n, dim=1)
    v = v.repeat_interleave(n, dim=1)
    return (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(-1) @ v

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 4, 3, 4, requires_grad=True), torch.randn(2, 2, 5, 4, requires_grad=True), torch.randn(2, 2, 5, 4, requires_grad=True))
    print(solve(*args))

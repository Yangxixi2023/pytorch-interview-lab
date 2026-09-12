"""Softmax 注意力

输入：
q: 浮点 Tensor [B, Q, D] — 查询。
k: 浮点 Tensor [B, K, D] — 键；K 可以不等于 Q。
v: 浮点 Tensor [B, K, Dv] — 值；Dv 可以不等于 D。

返回（多项按元组顺序）：
output: 浮点 Tensor [B, Q, Dv] — 非因果 scaled dot-product attention，保留 q/k/v 梯度。
"""

import math
import torch

def solve(q, k, v):
    return (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(-1) @ v

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 4, requires_grad=True), torch.randn(2, 5, 4, requires_grad=True), torch.randn(2, 5, 6, requires_grad=True))
    print(solve(*args))

"""多头交叉注意力

输入：
q: 浮点 Tensor [B, H, T, D] — 已投影的 query。
k: 浮点 Tensor [B, H, S, D] — 已投影的 key，S 可以不等于 T。
v: 浮点 Tensor [B, H, S, D] — 已投影的 value。

返回（多项按元组顺序）：
output: 浮点 Tensor [B, H, T, D] — 非因果注意力；不合并 head 维。
"""

import math
import torch

def solve(q, k, v):
    return (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(-1) @ v

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 4, 5, requires_grad=True), torch.randn(2, 3, 7, 5, requires_grad=True), torch.randn(2, 3, 7, 5, requires_grad=True))
    print(solve(*args))

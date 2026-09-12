"""张量并行 MLP（单机模拟）

输入：
x: 浮点 Tensor [..., D] — 输入特征。
up_shards: list[Tensor] 第 i 项 [D, Fi] — 列并行上投影，分片宽度可以不同。
down_shards: list[Tensor] 第 i 项 [Fi, O] — 与 up_shards 一一配对的下投影。

返回（多项按元组顺序）：
output: 浮点 Tensor [..., O] — 各分片 ReLU(x@up_i)@down_i 的和。
"""

import torch

def solve(x, up_shards, down_shards):
    return sum((torch.relu(x @ u) @ d for u, d in zip(up_shards, down_shards)))

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 4, requires_grad=True), [torch.randn(4, 2, requires_grad=True), torch.randn(4, 3, requires_grad=True)], [torch.randn(2, 5, requires_grad=True), torch.randn(3, 5, requires_grad=True)])
    print(solve(*args))

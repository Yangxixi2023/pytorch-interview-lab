"""INT8 量化线性层

输入：
x: 浮点 Tensor [..., D] — 输入特征。
weight: 浮点 Tensor [O, D] — 待量化权重，每输出行一个 scale。

返回（多项按元组顺序）：
output: 浮点 Tensor [..., O] — 使用反量化权重计算的线性输出。
q: int8 Tensor [O, D] — round 后限制到 [-127,127]，必须返回 int8。
scale: 浮点 Tensor [O, 1] — 逐行量化比例；全零行固定用 1。
"""

import torch

def solve(x, weight):
    scale = weight.abs().amax(-1, keepdim=True) / 127
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (weight / scale).round().clamp(-127, 127).to(torch.int8)
    return (x @ (q.to(x.dtype) * scale).T, q, scale)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 5), torch.randn(3, 5))
    print(solve(*args))

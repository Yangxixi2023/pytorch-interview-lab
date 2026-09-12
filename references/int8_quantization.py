"""INT8 量化线性层

输入：
x: 浮点 Tensor [..., D]
    输入特征。
weight: 浮点 Tensor [O, D]
    待量化权重，每输出行一个 scale。

返回：
output: 浮点 Tensor [..., O]
    使用反量化权重计算的线性输出。
q: int8 Tensor [O, D]
    round 后限制到 [-127,127]，必须返回 int8。
scale: 浮点 Tensor [O, 1]
    逐行量化比例；全零行固定用 1。
多个返回值按上面顺序组成元组。
"""

import torch


def solve(x, weight):
    row_maximum = weight.abs().amax(dim=-1, keepdim=True)
    scale = row_maximum / 127.0
    # 题目规定全零行 scale=1，量化结果仍然是全零。
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    quantized_weight = (weight / scale).round()
    quantized_weight = quantized_weight.clamp(-127, 127).to(torch.int8)
    # 实際矩阵乘法使用反量化后的浮点权重，而不是直接乘 int8。
    reconstructed_weight = quantized_weight.to(x.dtype) * scale
    output = x @ reconstructed_weight.transpose(-2, -1)
    return output, quantized_weight, scale


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 5), torch.randn(3, 5))
    print(solve(*args))

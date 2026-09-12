"""QLoRA 量化基座与低秩更新

输入：
x: 浮点 Tensor [..., D]
    输入特征，保留梯度。
codes: int64 Tensor [O, D]
    0..15 的给定量化码。
codebook: 浮点 Tensor [16]
    输入提供的码本，视为常量。
scales: 浮点 Tensor [O, 1]
    基座逐行缩放，视为常量。
a: 浮点 Tensor [r, D]
    可训练的低秩降维矩阵。
b: 浮点 Tensor [O, r]
    可训练的低秩升维矩阵。
alpha: float —
    低秩分支的缩放分子，实际系数 alpha/r。

返回：
output: 浮点 Tensor [..., O]
    量化基座输出加 LoRA 更新；只对 x/a/b 保留梯度。
"""

import torch


def solve(x, codes, codebook, scales, a, b, alpha):
    # codes 是码本索引；反量化的基座权重冻结。
    reconstructed_weight = codebook[codes] * scales
    reconstructed_weight = reconstructed_weight.detach()
    base_output = x @ reconstructed_weight.transpose(-2, -1)

    rank = a.shape[0]
    low_rank_features = x @ a.transpose(-2, -1)
    low_rank_update = low_rank_features @ b.transpose(-2, -1)
    output = base_output + (alpha / rank) * low_rank_update
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(2, 4, requires_grad=True),
        torch.randint(16, (3, 4)),
        torch.linspace(-1, 1, 16),
        torch.rand(3, 1),
        torch.randn(2, 4, requires_grad=True),
        torch.randn(3, 2, requires_grad=True),
        4.0,
    )
    print(solve(*args))

"""LoRA 低秩适配

输入：
x: 浮点 Tensor [..., Din] — 输入特征。
weight: 浮点 Tensor [Dout, Din] — 冻结的基座权重，必须 detach。
a: 浮点 Tensor [r, Din] — 低秩降维权重。
b: 浮点 Tensor [Dout, r] — 低秩升维权重。
alpha: float — — 低秩分支缩放分子，实际系数 alpha/r。

返回（多项按元组顺序）：
output: 浮点 Tensor [..., Dout] — 基座输出加低秩更新；x/a/b 保留梯度，weight 不求导。
"""

import torch

def solve(x, weight, a, b, alpha):
    return x @ weight.detach().T + alpha / a.shape[0] * (x @ a.T @ b.T)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 4, requires_grad=True), torch.randn(5, 4, requires_grad=True), torch.randn(2, 4, requires_grad=True), torch.randn(5, 2, requires_grad=True), 4.0)
    print(solve(*args))

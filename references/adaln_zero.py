"""自适应 LayerNorm Zero

输入：
x: 浮点 Tensor [B, T, D] — 待归一化激活。
residual: 浮点 Tensor [B, T, D] — 残差分支输入。
shift: 浮点 Tensor [B, D] — 每样本特征偏移。
scale: 浮点 Tensor [B, D] — 每样本特征缩放增量。
gate: 浮点 Tensor [B, D] — 残差更新的门控。
eps: float — — LayerNorm 稳定项。

返回（多项按元组顺序）：
output: 浮点 Tensor [B, T, D] — 门控调制后的残差输出；gate=0 时等于 residual。
"""

import torch
import torch.nn.functional as F

def solve(x, residual, shift, scale, gate, eps=1e-06):
    z = F.layer_norm(x, (x.shape[-1],), eps=eps)
    return residual + gate[:, None] * ((1 + scale[:, None]) * z + shift[:, None])

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 4, requires_grad=True), torch.randn(2, 3, 4, requires_grad=True)) + tuple((torch.randn(2, 4, requires_grad=True) for _ in range(3)))
    print(solve(*args))

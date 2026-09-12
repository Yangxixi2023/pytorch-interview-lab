"""深度可分离卷积

输入：
x: 浮点 Tensor [N, C, H, W] — 输入图像。
depthwise: 浮点 Tensor [C, 1, K, K] — K 为奇数，逐通道卷积，padding=K//2。
pointwise: 浮点 Tensor [O, C, 1, 1] — 逐点 1×1 通道混合核。

返回（多项按元组顺序）：
output: 浮点 Tensor [N, O, H, W] — 深度卷积后做逐点卷积，空间大小不变，无 bias。
"""

import torch
import torch.nn.functional as F

def solve(x, depthwise, pointwise):
    k = depthwise.shape[-1]
    p = k // 2
    windows = F.pad(x, (p, p, p, p)).unfold(2, k, 1).unfold(3, k, 1)
    z = torch.einsum('nchwij,cij->nchw', windows, depthwise[:, 0])
    return torch.einsum('nchw,oc->nohw', z, pointwise[:, :, 0, 0])

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 4, 4, requires_grad=True), torch.randn(3, 1, 3, 3, requires_grad=True), torch.randn(5, 3, 1, 1, requires_grad=True))
    print(solve(*args))

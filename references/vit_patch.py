"""ViT Patch Embedding

输入：
x: 浮点 Tensor [N, C, H, W] — H、W 都可被 patch 边长 p 整除。
kernel_size: int — — patch 边长 p。
stride: int — — 本题等于 kernel_size。

返回（多项按元组顺序）：
patches: 浮点 Tensor [N, (H/p)*(W/p), C*p*p] — patch 先行后列；每个 patch 内按 C、ph、pw 展平，不含投影。
"""

import torch

def solve(x, kernel_size, stride):
    n, c, h, w = x.shape
    p = kernel_size
    return x.unfold(2, p, stride).unfold(3, p, stride).permute(0, 2, 3, 1, 4, 5).reshape(n, -1, c * p * p)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.arange(96.0).reshape(2, 3, 4, 4).requires_grad_(), 2, 2)
    print(solve(*args))

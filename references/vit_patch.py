"""ViT Patch Embedding

输入：
x: 浮点 Tensor [N, C, H, W]
    H、W 都可被 patch 边长 p 整除。
kernel_size: int —
    patch 边长 p。
stride: int —
    本题等于 kernel_size。

返回：
patches: 浮点 Tensor [N, (H/p)*(W/p), C*p*p]
    patch 先行后列；每个 patch 内按 C、ph、pw 展平，不含投影。
"""

import torch


def solve(x, kernel_size, stride):
    batch_size, channels, height, width = x.shape
    patch_size = kernel_size
    patches = x.unfold(2, patch_size, stride)
    patches = patches.unfold(3, patch_size, stride)

    # [N,C,Hp,Wp,p,p] -> [N,Hp,Wp,C,p,p]，先行后列列出 patch。
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    patch_features = channels * patch_size * patch_size
    output = patches.reshape(batch_size, -1, patch_features)
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.arange(96.0).reshape(2, 3, 4, 4).requires_grad_(), 2, 2)
    print(solve(*args))

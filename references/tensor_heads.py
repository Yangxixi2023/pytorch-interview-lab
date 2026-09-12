"""张量变换与多头重排

输入：
x: 浮点 Tensor [B, T, D] — 可能不是连续张量，D 可被 heads 整除。
heads: int — — head 数 H，正整数。

返回（多项按元组顺序）：
split_heads: 浮点 Tensor [B, H, T, D/H] — 拆头并交换时间与头维度，保留梯度。
merged: 浮点 Tensor [B, T, D] — 合并还原的张量，值与输入一致。
"""

import torch

def solve(x, heads):
    b, t, d = x.shape
    split = x.reshape(b, t, heads, d // heads).transpose(1, 2)
    return (split, split.transpose(1, 2).reshape(b, t, d))

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.arange(48.0).reshape(2, 2, 12).requires_grad_(), 3)
    print(solve(*args))

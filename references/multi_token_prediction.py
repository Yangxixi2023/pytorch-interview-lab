"""多 Token 预测

输入：
logits: 浮点 Tensor [K, B, T, V] — 第 k 个 head 在 t 位置预测 t+k+1；k 从0开始。
tokens: int64 Tensor [B, T] — 目标 token ID；保证 T>K。

返回（多项按元组顺序）：
loss: 浮点 Tensor [] — 零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch
import torch.nn.functional as F

def solve(logits, tokens):
    loss = logits.new_zeros(())
    count = 0
    for k in range(logits.shape[0]):
        target = tokens[:, k + 1:]
        loss = loss + F.cross_entropy(logits[k, :, :-k - 1].reshape(-1, logits.shape[-1]), target.reshape(-1), reduction='sum')
        count += target.numel()
    return loss / count

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(3, 2, 5, 7, requires_grad=True), torch.randint(7, (2, 5)))
    print(solve(*args))

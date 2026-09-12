"""对比损失 InfoNCE

输入：
queries: 浮点 Tensor [N, D] — 需要在最后一维 L2 normalize。
keys: 浮点 Tensor [N, D] — 第 i 个 key 是第 i 个 query 的正样本。
temperature: float — — 正的温度系数。

返回（多项按元组顺序）：
loss: 浮点 Tensor [] — 零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch
import torch.nn.functional as F

def solve(queries, keys, temperature=0.1):
    scores = F.normalize(queries, dim=-1) @ F.normalize(keys, dim=-1).T / temperature
    return F.cross_entropy(scores, torch.arange(scores.shape[0], device=scores.device))

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(4, 5, requires_grad=True), torch.randn(4, 5, requires_grad=True))
    print(solve(*args))

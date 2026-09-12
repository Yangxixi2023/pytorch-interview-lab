"""MoE 负载均衡损失

输入：
router_logits: 浮点 Tensor [T, E] — 每 token 对每 expert 的未归一化分数。

返回（多项按元组顺序）：
loss: 浮点 Tensor [] — 零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch
import torch.nn.functional as F

def solve(router_logits):
    p = router_logits.softmax(-1)
    e = p.shape[-1]
    f = F.one_hot(p.argmax(-1), e).to(p.dtype).mean(0)
    return e * (f * p.mean(0)).sum()

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(8, 3, requires_grad=True),)
    print(solve(*args))

"""混合专家 MoE

输入：
x: 浮点 Tensor [T, D] — token 特征。
router_logits: 浮点 Tensor [T, E] — 专家路由分数，测试不含并列。
weights: 浮点 Tensor [E, D, O] — 每个专家的线性层权重。
top_k: int — — 每 token 选择的专家数，1<=top_k<=E。

返回（多项按元组顺序）：
output: 浮点 Tensor [T, O] — 只在选中专家 logits 内归一化，并加权聚合输出。
"""

import torch

def solve(x, router_logits, weights, top_k=2):
    scores, ids = router_logits.topk(top_k, dim=-1)
    p = scores.softmax(-1)
    y = torch.einsum('td,tkdo->tko', x, weights[ids])
    return (y * p[..., None]).sum(1)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(5, 3, requires_grad=True), torch.randn(5, 4, requires_grad=True), torch.randn(4, 3, 2, requires_grad=True), 2)
    print(solve(*args))

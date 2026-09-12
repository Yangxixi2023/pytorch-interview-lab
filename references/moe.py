"""混合专家 MoE

输入：
x: 浮点 Tensor [T, D]
    token 特征。
router_logits: 浮点 Tensor [T, E]
    专家路由分数，测试不含并列。
weights: 浮点 Tensor [E, D, O]
    每个专家的线性层权重。
top_k: int —
    每 token 选择的专家数，1<=top_k<=E。

返回：
output: 浮点 Tensor [T, O]
    只在选中专家 logits 内归一化，并加权聚合输出。
"""

import torch


def solve(x, router_logits, weights, top_k=2):
    # 只在选中的 top-k logits 内归一化，不使用全专家 softmax 权重。
    selected_logits, selected_experts = router_logits.topk(top_k, dim=-1)
    routing_weights = selected_logits.softmax(dim=-1)  # [T,K]
    selected_weights = weights[selected_experts]  # [T,K,D,O]

    expert_outputs = torch.einsum("td,tkdo->tko", x, selected_weights)
    weighted_outputs = expert_outputs * routing_weights.unsqueeze(-1)
    output = weighted_outputs.sum(dim=1)
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(5, 3, requires_grad=True),
        torch.randn(5, 4, requires_grad=True),
        torch.randn(4, 3, 2, requires_grad=True),
        2,
    )
    print(solve(*args))

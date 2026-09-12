"""MoE 负载均衡损失

输入：
router_logits: 浮点 Tensor [T, E]
    每 token 对每 expert 的未归一化分数。

返回：
loss: 浮点 Tensor []
    零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch
import torch.nn.functional as F


def solve(router_logits):
    routing_probabilities = router_logits.softmax(dim=-1)  # [T,E]
    expert_count = routing_probabilities.shape[-1]
    selected_experts = routing_probabilities.argmax(dim=-1)

    # 硬路由频率不求导；平均软概率保留对 router 的梯度。
    assignments = F.one_hot(selected_experts, num_classes=expert_count)
    expert_frequency = assignments.to(routing_probabilities.dtype).mean(
        dim=0
    )
    mean_probability = routing_probabilities.mean(dim=0)
    loss = expert_count * (expert_frequency * mean_probability).sum()
    return loss


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(8, 3, requires_grad=True),)
    print(solve(*args))

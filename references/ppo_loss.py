"""PPO 损失

输入：
new_logp: 浮点 Tensor [B, T]
    当前策略对已采样 token 的自然对数概率；需要保留梯度。
old_logp: 浮点 Tensor [B, T]
    采样时旧策略的 log probability；计算时 detach。
advantages: 浮点 Tensor [B, T]
    逐 token 优势，可正可负；计算时 detach。
mask: bool Tensor [B, T]
    True 为有效 token，False 为 padding；整个 batch 至少一个 True。
clip_eps: float —
    对称裁剪半径；ratio 裁剪到 [1-clip_eps, 1+clip_eps]。

返回：
loss: 浮点 Tensor []
    零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch


def solve(new_logp, old_logp, advantages, mask, clip_eps=0.2):
    # 旧策略和优势是本轮更新的固定目标。
    old_logp = old_logp.detach()
    advantages = advantages.detach()

    probability_ratio = torch.exp(new_logp - old_logp)  # [B, T]
    clipped_ratio = torch.clamp(
        probability_ratio, min=1 - clip_eps, max=1 + clip_eps
    )

    # 对正负优势都取较保守的 surrogate，而非只裁剪 ratio。
    unclipped_objective = probability_ratio * advantages
    clipped_objective = clipped_ratio * advantages
    token_objective = torch.minimum(unclipped_objective, clipped_objective)

    # 只按有效 token 平均，返回零维 Tensor，保留当前策略梯度。
    valid_objective = token_objective[mask]
    loss = -valid_objective.mean()
    return loss


if __name__ == "__main__":
    torch.manual_seed(17)
    new = torch.tensor([[0.0, 0.7, -0.7, 0.1]], requires_grad=True)
    args = (
        new,
        torch.zeros_like(new),
        torch.tensor([[1.0, 1.0, -1.0, -2.0]]),
        torch.ones_like(new, dtype=torch.bool),
    )
    print(solve(*args))

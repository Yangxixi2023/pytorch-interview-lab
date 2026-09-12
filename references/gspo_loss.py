"""GSPO 损失

输入：
new_logp: 浮点 Tensor [B, T]
    当前策略的 token log probability；保留梯度。
old_logp: 浮点 Tensor [B, T]
    旧策略 token log probability；detach 后使用。
advantages: 浮点 Tensor [B]
    每条序列一个已标准化优势；不是 [B,T]；detach 后使用。
mask: bool Tensor [B, T]
    True 为有效 token；每条序列至少一个 True。
clip_low: float —
    ratio 的下界为 1-clip_low。
clip_high: float —
    ratio 的上界为 1+clip_high。

返回：
loss: 浮点 Tensor []
    零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch


def solve(
    new_logp, old_logp, advantages, mask, clip_low=0.2, clip_high=0.28
):
    old_logp = old_logp.detach()
    sequence_advantages = advantages.detach()  # [B]
    sequence_lengths = mask.sum(dim=-1)  # [B]，题目保证非零

    # 先平均 log ratio，再取 exp；不是 token ratio 的算术平均。
    token_log_ratio = new_logp - old_logp
    valid_log_ratio = token_log_ratio.masked_fill(~mask, 0.0)
    mean_log_ratio = valid_log_ratio.sum(dim=-1) / sequence_lengths
    sequence_ratio = torch.exp(mean_log_ratio)

    clipped_ratio = sequence_ratio.clamp(1 - clip_low, 1 + clip_high)
    unclipped_objective = sequence_ratio * sequence_advantages
    clipped_objective = clipped_ratio * sequence_advantages
    sequence_objective = torch.minimum(
        unclipped_objective, clipped_objective
    )

    # 每条序列权重相同，与其有效 token 数无关。
    loss = -sequence_objective.mean()
    return loss


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.tensor(
            [[0.1, 0.4, -0.4], [0.5, -0.5, 0.1]], requires_grad=True
        ),
        torch.zeros(2, 3),
        torch.tensor([1.0, -1.0]),
        torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool),
    )
    print(solve(*args))

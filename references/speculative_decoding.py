"""推测解码接受步骤

输入：
draft_tokens: int64 Tensor [K]
    草稿 token ID。
draft_probs: 浮点 Tensor [K, V]
    草稿模型的已归一化概率，候选 token 概率>0。
target_probs: 浮点 Tensor [K, V]
    目标模型的已归一化概率。
uniforms: 浮点 Tensor [K]
    提供的 [0,1) 随机数，不要重新采样。

返回：
accepted: list[int] 长度 0..K
    从前向后已接受的草稿 token。
residual: Tensor 或 None [V] 或 —
    首次拒绝时返回 normalize(max(p-q,0))；全部接受返回 None。
多个返回值按上面顺序组成元组。
"""

import torch


def solve(draft_tokens, draft_probs, target_probs, uniforms):
    accepted_tokens = []
    for position, token in enumerate(draft_tokens.tolist()):
        target_probability = target_probs[position, token]
        draft_probability = draft_probs[position, token]
        acceptance_ratio = float(target_probability / draft_probability)
        acceptance_probability = min(1.0, acceptance_ratio)

        if float(uniforms[position]) >= acceptance_probability:
            # 首次拒绝后停止；剩余分布仅保留目标比草稿多出的概率质量。
            residual = target_probs[position] - draft_probs[position]
            residual = residual.clamp_min(0.0)
            residual = residual / residual.sum()
            return accepted_tokens, residual

        accepted_tokens.append(token)

    return accepted_tokens, None


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.tensor([0, 1]),
        torch.tensor([[0.5, 0.5], [0.1, 0.9]]),
        torch.tensor([[0.7, 0.3], [0.8, 0.2]]),
        torch.tensor([0.2, 0.8]),
    )
    print(solve(*args))

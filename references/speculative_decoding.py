"""推测解码接受步骤

输入：
draft_tokens: int64 Tensor [K] — 草稿 token ID。
draft_probs: 浮点 Tensor [K, V] — 草稿模型的已归一化概率，候选 token 概率>0。
target_probs: 浮点 Tensor [K, V] — 目标模型的已归一化概率。
uniforms: 浮点 Tensor [K] — 提供的 [0,1) 随机数，不要重新采样。

返回（多项按元组顺序）：
accepted: list[int] 长度 0..K — 从前向后已接受的草稿 token。
residual: Tensor 或 None [V] 或 — — 首次拒绝时返回 normalize(max(p-q,0))；全部接受返回 None。
"""

import torch

def solve(draft_tokens, draft_probs, target_probs, uniforms):
    accepted = []
    for i, token in enumerate(draft_tokens.tolist()):
        ratio = min(1.0, float(target_probs[i, token] / draft_probs[i, token]))
        if float(uniforms[i]) >= ratio:
            residual = (target_probs[i] - draft_probs[i]).clamp_min(0)
            return (accepted, residual / residual.sum())
        accepted.append(token)
    return (accepted, None)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.tensor([0, 1]), torch.tensor([[0.5, 0.5], [0.1, 0.9]]), torch.tensor([[0.7, 0.3], [0.8, 0.2]]), torch.tensor([0.2, 0.8]))
    print(solve(*args))

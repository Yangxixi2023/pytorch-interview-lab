"""Top-k / Top-p 采样分布

输入：
logits: 浮点 Tensor [V]
    一维词表分数，测试不含并列。
top_k: int —
    0 表示不做 top-k；否则 1..V。
top_p: float —
    累计概率阈值，0<top_p<=1。
temperature: float —
    严格正的温度。

返回：
probabilities: 浮点 Tensor [V]
    原词表顺序的归一化概率，被过滤项为0；返回分布，不采样 token。
"""

import torch


def solve(logits, top_k, top_p, temperature=1.0):
    scaled_logits = logits / temperature
    if top_k > 0:
        kept_logits, kept_indices = scaled_logits.topk(top_k)
        filtered_logits = torch.full_like(scaled_logits, float("-inf"))
        filtered_logits[kept_indices] = kept_logits
        scaled_logits = filtered_logits

    sorted_logits, sorted_indices = scaled_logits.sort(descending=True)
    sorted_probabilities = sorted_logits.softmax(dim=-1)
    # 保留首次让累计概率达到阈值的那个 token。
    probability_before_token = sorted_probabilities.cumsum(dim=-1)
    probability_before_token = (
        probability_before_token - sorted_probabilities
    )
    remove = probability_before_token >= top_p
    kept_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
    kept_probabilities = kept_probabilities / kept_probabilities.sum()

    output = torch.zeros_like(kept_probabilities)
    output[sorted_indices] = kept_probabilities  # 还原词表顺序。
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.tensor([0.1, 2.0, 1.0, -1.0, 0.5]), 3, 0.8, 0.7)
    print(solve(*args))

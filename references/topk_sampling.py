"""Top-k / Top-p 采样分布

输入：
logits: 浮点 Tensor [V] — 一维词表分数，测试不含并列。
top_k: int — — 0 表示不做 top-k；否则 1..V。
top_p: float — — 累计概率阈值，0<top_p<=1。
temperature: float — — 严格正的温度。

返回（多项按元组顺序）：
probabilities: 浮点 Tensor [V] — 原词表顺序的归一化概率，被过滤项为0；返回分布，不采样 token。
"""

import torch

def solve(logits, top_k, top_p, temperature=1.0):
    z = logits / temperature
    if top_k > 0:
        vals, ids = z.topk(top_k)
        filtered = torch.full_like(z, float('-inf'))
        filtered[ids] = vals
        z = filtered
    vals, ids = z.sort(descending=True)
    p = vals.softmax(-1)
    remove = p.cumsum(-1) - p >= top_p
    p = p.masked_fill(remove, 0)
    p = p / p.sum()
    out = torch.zeros_like(p)
    out[ids] = p
    return out

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.tensor([0.1, 2.0, 1.0, -1.0, 0.5]), 3, 0.8, 0.7)
    print(solve(*args))

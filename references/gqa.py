"""分组查询注意力 GQA

输入：
q: 浮点 Tensor [B, Hq, T, D]
    Hq 必须可被 Hkv 整除。
k: 浮点 Tensor [B, Hkv, S, D]
    连续 Hq/Hkv 个 query head 共用一个 KV head。
v: 浮点 Tensor [B, Hkv, S, D]
    与 key 的头数和序列长度一致。

返回：
output: 浮点 Tensor [B, Hq, T, D]
    非因果 GQA 输出，不合并 head 维。
"""

import math
import torch


def solve(q, k, v):
    query_heads = q.shape[1]
    kv_heads = k.shape[1]
    queries_per_kv_head = query_heads // kv_heads

    # 相邻 query head 共用一个 KV head，不能用普通 repeat 交错复制。
    expanded_key = k.repeat_interleave(queries_per_kv_head, dim=1)
    expanded_value = v.repeat_interleave(queries_per_kv_head, dim=1)

    scores = q @ expanded_key.transpose(-2, -1)
    scores = scores / math.sqrt(q.shape[-1])
    attention_weights = torch.softmax(scores, dim=-1)
    output = attention_weights @ expanded_value
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(2, 4, 3, 4, requires_grad=True),
        torch.randn(2, 2, 5, 4, requires_grad=True),
        torch.randn(2, 2, 5, 4, requires_grad=True),
    )
    print(solve(*args))

"""KV Cache 注意力

输入：
q: 浮点 Tensor [B, H, 1, D]
    当前解码步的 query。
new_k: 浮点 Tensor [B, H, 1, D]
    当前新增 key。
new_v: 浮点 Tensor [B, H, 1, D]
    当前新增 value。
cached_k: 浮点 Tensor [B, H, S, D]
    历史 key，可为空 S=0。
cached_v: 浮点 Tensor [B, H, S, D]
    历史 value，与 cached_k 长度相同。

返回：
output: 浮点 Tensor [B, H, 1, D]
    当前 query 对所有已缓存和新增 KV 的注意力。
updated_k: 浮点 Tensor [B, H, S+1, D]
    追加后的 key，不修改原输入。
updated_v: 浮点 Tensor [B, H, S+1, D]
    追加后的 value，不修改原输入。
多个返回值按上面顺序组成元组。
"""

import math
import torch


def solve(q, new_k, new_v, cached_k, cached_v):
    # 时间维是倒数第二维，允许历史缓存长度为0。
    updated_key = torch.cat((cached_k, new_k), dim=-2)
    updated_value = torch.cat((cached_v, new_v), dim=-2)

    scores = q @ updated_key.transpose(-2, -1)
    scores = scores / math.sqrt(q.shape[-1])
    attention_weights = scores.softmax(dim=-1)
    output = attention_weights @ updated_value
    return output, updated_key, updated_value


if __name__ == "__main__":
    torch.manual_seed(17)
    args = tuple(
        (torch.randn(1, 2, 1, 3, requires_grad=True) for _ in range(3))
    ) + tuple((torch.randn(1, 2, 4, 3) for _ in range(2)))
    print(solve(*args))

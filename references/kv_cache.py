"""KV Cache 注意力

输入：
q: 浮点 Tensor [B, H, 1, D] — 当前解码步的 query。
new_k: 浮点 Tensor [B, H, 1, D] — 当前新增 key。
new_v: 浮点 Tensor [B, H, 1, D] — 当前新增 value。
cached_k: 浮点 Tensor [B, H, S, D] — 历史 key，可为空 S=0。
cached_v: 浮点 Tensor [B, H, S, D] — 历史 value，与 cached_k 长度相同。

返回（多项按元组顺序）：
output: 浮点 Tensor [B, H, 1, D] — 当前 query 对所有已缓存和新增 KV 的注意力。
updated_k: 浮点 Tensor [B, H, S+1, D] — 追加后的 key，不修改原输入。
updated_v: 浮点 Tensor [B, H, S+1, D] — 追加后的 value，不修改原输入。
"""

import math
import torch

def solve(q, new_k, new_v, cached_k, cached_v):
    k = torch.cat((cached_k, new_k), dim=-2)
    v = torch.cat((cached_v, new_v), dim=-2)
    return ((q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(-1) @ v, k, v)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = tuple((torch.randn(1, 2, 1, 3, requires_grad=True) for _ in range(3))) + tuple((torch.randn(1, 2, 4, 3) for _ in range(2)))
    print(solve(*args))

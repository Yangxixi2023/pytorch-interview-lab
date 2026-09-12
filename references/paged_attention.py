"""分页注意力

输入：
q: 浮点 Tensor [D]
    单序列单步 query。
key_pages: 浮点 Tensor [P, page_size, D]
    物理 key 页。
value_pages: 浮点 Tensor [P, page_size, D]
    物理 value 页。
block_table: int64 Tensor [num_blocks]
    按逻辑序列顺序列出的物理页索引。
length: int —
    有效 KV token 数，>=1；末页 padding 不计入。

返回：
output: 浮点 Tensor [D]
    按逻辑顺序收集前 length 个 KV 后的注意力输出。
"""

import math
import torch


def solve(q, key_pages, value_pages, block_table, length):
    # block_table 给出逻辑顺序，物理页在内存中可以不连续。
    ordered_key_pages = key_pages[block_table]
    ordered_value_pages = value_pages[block_table]
    keys = ordered_key_pages.flatten(0, 1)[:length]
    values = ordered_value_pages.flatten(0, 1)[:length]

    # 截取有效长度后，末页 padding 不会参与 softmax。
    scores = (q @ keys.transpose(0, 1)) / math.sqrt(q.shape[-1])
    attention_weights = scores.softmax(dim=-1)
    output = attention_weights @ values
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(4, requires_grad=True),
        torch.randn(4, 3, 4, requires_grad=True),
        torch.randn(4, 3, 4, requires_grad=True),
        torch.tensor([2, 0, 3]),
        7,
    )
    print(solve(*args))

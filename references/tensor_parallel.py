"""张量并行 MLP（单机模拟）

输入：
x: 浮点 Tensor [..., D]
    输入特征。
up_shards: list[Tensor] 第 i 项 [D, Fi]
    列并行上投影，分片宽度可以不同。
down_shards: list[Tensor] 第 i 项 [Fi, O]
    与 up_shards 一一配对的下投影。

返回：
output: 浮点 Tensor [..., O]
    各分片 ReLU(x@up_i)@down_i 的和。
"""

import torch


def solve(x, up_shards, down_shards):
    partial_outputs = []
    for up_weight, down_weight in zip(up_shards, down_shards):
        local_hidden = torch.relu(x @ up_weight)
        local_output = local_hidden @ down_weight
        partial_outputs.append(local_output)

    # 行并行下投影之后应求和，而不是沿特征维拼接。
    output = sum(partial_outputs)
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(2, 4, requires_grad=True),
        [
            torch.randn(4, 2, requires_grad=True),
            torch.randn(4, 3, requires_grad=True),
        ],
        [
            torch.randn(2, 5, requires_grad=True),
            torch.randn(3, 5, requires_grad=True),
        ],
    )
    print(solve(*args))

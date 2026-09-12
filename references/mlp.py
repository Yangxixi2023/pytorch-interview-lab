"""SwiGLU MLP

输入：
x: 浮点 Tensor [..., D]
    输入特征。
gate_weight: 浮点 Tensor [D, F]
    门控投影。
up_weight: 浮点 Tensor [D, F]
    上投影。
down_weight: 浮点 Tensor [F, D]
    下投影。

返回：
output: 浮点 Tensor 与 x 相同
    (SiLU(x@gate_weight)*(x@up_weight))@down_weight。
"""

import torch


def solve(x, gate_weight, up_weight, down_weight):
    gate_input = x @ gate_weight
    value_features = x @ up_weight
    # gate 分支控制信息通过，up 分支携带值；两者逐元素相乘。
    activated_gate = gate_input * torch.sigmoid(gate_input)
    hidden_features = activated_gate * value_features
    output = hidden_features @ down_weight
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(2, 3, 4, requires_grad=True),
        torch.randn(4, 6, requires_grad=True),
        torch.randn(4, 6, requires_grad=True),
        torch.randn(6, 4, requires_grad=True),
    )
    print(solve(*args))

"""Mamba SSM 步骤

输入：
u: 浮点 Tensor [B, T, D]
    序列输入。
delta: 浮点 Tensor [B, T, D]
    每步离散化步长。
a: 浮点 Tensor [D, N]
    状态转移参数。
b: 浮点 Tensor [B, T, N]
    选择性输入参数。
c: 浮点 Tensor [B, T, N]
    选择性读出参数。
d: 浮点 Tensor [D]
    输入跳连系数。
initial: 浮点 Tensor [B, D, N]
    初始状态，可以非零。

返回：
y: 浮点 Tensor [B, T, D]
    每步状态读出加输入跳连。
final_h: 浮点 Tensor [B, D, N]
    最后一个时间步之后的状态。
多个返回值按上面顺序组成元组。
"""

import torch


def solve(u, delta, a, b, c, d, initial):
    state = initial  # [B,D,N]
    outputs = []

    for time in range(u.shape[1]):
        step_size = delta[:, time, :, None]  # [B,D,1]
        transition = torch.exp(step_size * a)
        input_coefficients = b[:, time, None, :]  # [B,1,N]
        current_input = u[:, time, :, None]  # [B,D,1]

        # 本题输入项采用 delta*B*u 离散化。
        input_update = step_size * input_coefficients * current_input
        state = transition * state + input_update

        readout_coefficients = c[:, time, None, :]
        state_output = (state * readout_coefficients).sum(dim=-1)
        direct_output = d * u[:, time, :]
        outputs.append(state_output + direct_output)

    sequence_output = torch.stack(outputs, dim=1)
    return sequence_output, state


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(2, 4, 3, requires_grad=True),
        torch.rand(2, 4, 3),
        -torch.rand(3, 2),
        torch.randn(2, 4, 2),
        torch.randn(2, 4, 2),
        torch.randn(3),
        torch.zeros(2, 3, 2),
    )
    print(solve(*args))

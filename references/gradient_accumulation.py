"""梯度累积

输入：
x: 浮点 Tensor [N, D]
    输入特征。
y: 浮点 Tensor [N, O]
    回归目标。
weight: 浮点 Tensor [D, O]
    线性映射参数。
microbatch_size: int —
    正整数；最后一个 microbatch 可以不足此大小。

返回：
grad_weight: 浮点 Tensor [D, O]
    全 batch、所有输出元素的 mean MSE 对 weight 的手动梯度；不要用 autograd。
"""

import torch


def solve(x, y, weight, microbatch_size):
    weight_gradient = torch.zeros_like(weight)
    total_output_elements = y.numel()

    for start in range(0, x.shape[0], microbatch_size):
        microbatch_x = x[start : start + microbatch_size]
        microbatch_y = y[start : start + microbatch_size]
        prediction_error = microbatch_x @ weight - microbatch_y

        # 每块都除以全 batch 的元素数，尾块较小时也不会权重过大。
        prediction_gradient = 2.0 * prediction_error / total_output_elements
        microbatch_gradient = (
            microbatch_x.transpose(0, 1) @ prediction_gradient
        )
        weight_gradient = weight_gradient + microbatch_gradient

    return weight_gradient


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(7, 3), torch.randn(7, 2), torch.randn(3, 2), 3)
    print(solve(*args))

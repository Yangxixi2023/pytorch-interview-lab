"""梯度累积

输入：
x: 浮点 Tensor [N, D] — 输入特征。
y: 浮点 Tensor [N, O] — 回归目标。
weight: 浮点 Tensor [D, O] — 线性映射参数。
microbatch_size: int — — 正整数；最后一个 microbatch 可以不足此大小。

返回（多项按元组顺序）：
grad_weight: 浮点 Tensor [D, O] — 全 batch、所有输出元素的 mean MSE 对 weight 的手动梯度；不要用 autograd。
"""

import torch

def solve(x, y, weight, microbatch_size):
    grad = torch.zeros_like(weight)
    for start in range(0, len(x), microbatch_size):
        xb = x[start:start + microbatch_size]
        yb = y[start:start + microbatch_size]
        grad += 2 * xb.T @ (xb @ weight - yb) / y.numel()
    return grad

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(7, 3), torch.randn(7, 2), torch.randn(3, 2), 3)
    print(solve(*args))

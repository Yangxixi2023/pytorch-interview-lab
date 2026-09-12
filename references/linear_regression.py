"""多元线性回归与反向传播

输入：
x: 浮点 Tensor [N, D] — 输入特征。
y: 浮点 Tensor [N, O] — 目标输出。
weight: 浮点 Tensor [D, O] — 线性回归权重。
bias: 浮点 Tensor [O] — 偏移。

返回（多项按元组顺序）：
loss: 浮点 Tensor [] — 所有 N*O 元素的 mean MSE。
dweight: 浮点 Tensor [D, O] — 手动推导的 weight 梯度，不用 autograd。
dbias: 浮点 Tensor [O] — 手动推导的 bias 梯度。
"""

import torch

def solve(x, y, weight, bias):
    err = x @ weight + bias - y
    return (err.square().mean(), 2 * x.T @ err / err.numel(), 2 * err.sum(0) / err.numel())

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(5, 3), torch.randn(5, 2), torch.randn(3, 2), torch.randn(2))
    print(solve(*args))

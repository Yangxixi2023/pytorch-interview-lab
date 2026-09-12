"""多元线性回归与反向传播

输入：
x: 浮点 Tensor [N, D]
    输入特征。
y: 浮点 Tensor [N, O]
    目标输出。
weight: 浮点 Tensor [D, O]
    线性回归权重。
bias: 浮点 Tensor [O]
    偏移。

返回：
loss: 浮点 Tensor []
    所有 N*O 元素的 mean MSE。
dweight: 浮点 Tensor [D, O]
    手动推导的 weight 梯度，不用 autograd。
dbias: 浮点 Tensor [O]
    手动推导的 bias 梯度。
多个返回值按上面顺序组成元组。
"""

import torch


def solve(x, y, weight, bias):
    predictions = x @ weight + bias  # [N,O]
    errors = predictions - y
    loss = errors.square().mean()

    # loss 对预测的导数；分母是全部 N*O 个元素。
    prediction_gradient = 2.0 * errors / errors.numel()
    weight_gradient = x.transpose(0, 1) @ prediction_gradient
    bias_gradient = prediction_gradient.sum(dim=0)
    return loss, weight_gradient, bias_gradient


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(5, 3),
        torch.randn(5, 2),
        torch.randn(3, 2),
        torch.randn(2),
    )
    print(solve(*args))

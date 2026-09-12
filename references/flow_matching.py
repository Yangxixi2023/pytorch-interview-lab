"""流匹配损失

输入：
pred_velocity: 浮点 Tensor 任意形状
    可训练的预测速度。
x0: 浮点 Tensor 与 pred_velocity 相同
    路径起点，视为常量。
x1: 浮点 Tensor 与 pred_velocity 相同
    路径终点，视为常量。

返回：
loss: 浮点 Tensor []
    零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch


def solve(pred_velocity, x0, x1):
    # 线性插值路径的速度是终点减起点，与时间 t 无关。
    target_velocity = (x1 - x0).detach()
    prediction_error = pred_velocity - target_velocity
    loss = prediction_error.square().mean()
    return loss


if __name__ == "__main__":
    torch.manual_seed(17)
    args = tuple(
        (torch.randn(2, 3, 4, requires_grad=True) for _ in range(3))
    )
    print(solve(*args))

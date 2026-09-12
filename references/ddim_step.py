"""DDIM 采样步骤

输入：
xt: 浮点 Tensor 任意形状
    时刻 t 的噪声样本。
eps_pred: 浮点 Tensor 与 xt 相同
    模型预测的噪声。
alpha_bar_t: float —
    当前累计 alpha，>0。
alpha_bar_prev: float —
    前一时刻累计 alpha，<=1。

返回：
x_prev: 浮点 Tensor 与 xt 相同
    eta=0 的确定性 DDIM 更新结果，不截断预测 x0。
"""

import math
import torch


def solve(xt, eps_pred, alpha_bar_t, alpha_bar_prev):
    current_signal_scale = math.sqrt(alpha_bar_t)
    current_noise_scale = math.sqrt(1.0 - alpha_bar_t)
    predicted_x0 = (
        xt - current_noise_scale * eps_pred
    ) / current_signal_scale

    # eta=0：不额外采样随机噪声，使用同一个预测噪声方向。
    previous_signal_scale = math.sqrt(alpha_bar_prev)
    previous_noise_scale = math.sqrt(1.0 - alpha_bar_prev)
    x_previous = previous_signal_scale * predicted_x0
    x_previous = x_previous + previous_noise_scale * eps_pred
    return x_previous


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 4), torch.randn(2, 3, 4), 0.4, 0.6)
    print(solve(*args))

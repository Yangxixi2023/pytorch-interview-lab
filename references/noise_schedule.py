"""扩散噪声调度

输入：
steps: int —
    正的总步数。
beta_start: float —
    第一个 beta。
beta_end: float —
    最后一个 beta（steps=1 时只用起点）。

返回：
betas: 浮点 Tensor [steps]
    float32 线性 beta 序列。
alphas: 浮点 Tensor [steps]
    1-betas。
alpha_bars: 浮点 Tensor [steps]
    alphas 的前缀连乘。
多个返回值按上面顺序组成元组。
"""

import torch


def solve(steps, beta_start, beta_end):
    betas = torch.linspace(beta_start, beta_end, steps)
    alphas = 1.0 - betas
    # 累计 alpha 决定从干净样本直接加噪到该时刻的比例。
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (5, 0.0001, 0.02)
    print(solve(*args))

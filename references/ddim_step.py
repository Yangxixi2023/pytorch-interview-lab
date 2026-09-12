"""DDIM 采样步骤

输入：
xt: 浮点 Tensor 任意形状 — 时刻 t 的噪声样本。
eps_pred: 浮点 Tensor 与 xt 相同 — 模型预测的噪声。
alpha_bar_t: float — — 当前累计 alpha，>0。
alpha_bar_prev: float — — 前一时刻累计 alpha，<=1。

返回（多项按元组顺序）：
x_prev: 浮点 Tensor 与 xt 相同 — eta=0 的确定性 DDIM 更新结果，不截断预测 x0。
"""

import math
import torch

def solve(xt, eps_pred, alpha_bar_t, alpha_bar_prev):
    x0 = (xt - math.sqrt(1 - alpha_bar_t) * eps_pred) / math.sqrt(alpha_bar_t)
    return math.sqrt(alpha_bar_prev) * x0 + math.sqrt(1 - alpha_bar_prev) * eps_pred

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 3, 4), torch.randn(2, 3, 4), 0.4, 0.6)
    print(solve(*args))

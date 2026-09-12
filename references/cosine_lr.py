"""余弦学习率（含预热）

输入：
step: int — — 当前步，0<=step<=total_steps。
warmup_steps: int — — 预热步数，0<=warmup_steps<total_steps。
total_steps: int — — 总步数。
max_lr: float — — 预热结束时的学习率。
min_lr: float — — 余弦阶段终点学习率。

返回（多项按元组顺序）：
learning_rate: float — — 当前步的 Python 浮点学习率，不返回 Tensor。
"""

import math
import torch

def solve(step, warmup_steps, total_steps, max_lr, min_lr=0.0):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    p = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (2, 10, 100, 0.01, 0.001)
    print(solve(*args))

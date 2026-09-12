"""激活检查点

输入：
x: 浮点 Tensor [..., D] — MLP 输入；可能不需要梯度。
w1: 浮点 Tensor [D, F] — 第一层可训练权重。
w2: 浮点 Tensor [F, O] — 第二层可训练权重。

返回（多项按元组顺序）：
output: 浮点 Tensor [..., O] — checkpoint 包裹的 ReLU(x@w1)@w2，保留参数梯度。
"""

import torch

def solve(x, w1, w2):
    from torch.utils.checkpoint import checkpoint
    return checkpoint(lambda a, b, c: torch.relu(a @ b) @ c, x, w1, w2, use_reentrant=False)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 4, requires_grad=True), torch.randn(4, 6, requires_grad=True), torch.randn(6, 3, requires_grad=True))
    print(solve(*args))

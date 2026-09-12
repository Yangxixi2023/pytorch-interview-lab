"""DPO 损失

输入：
chosen: 浮点 Tensor [B] — 当前策略对偏好回答的整序列 log probability 之和；保留梯度。
rejected: 浮点 Tensor [B] — 当前策略对非偏好回答的整序列 log probability 之和；保留梯度。
ref_chosen: 浮点 Tensor [B] — 参考策略偏好回答 log probability；detach。
ref_rejected: 浮点 Tensor [B] — 参考策略非偏好回答 log probability；detach。
beta: float — — 偏好 margin 的缩放系数。

返回（多项按元组顺序）：
loss: 浮点 Tensor [] — 零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch
import torch.nn.functional as F

def solve(chosen, rejected, ref_chosen, ref_rejected, beta=0.1):
    return -F.logsigmoid(beta * (chosen - rejected - (ref_chosen.detach() - ref_rejected.detach()))).mean()

if __name__ == "__main__":
    torch.manual_seed(17)
    args = tuple((torch.randn(5, requires_grad=True) for _ in range(4)))
    print(solve(*args))

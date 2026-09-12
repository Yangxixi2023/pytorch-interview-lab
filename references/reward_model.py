"""Bradley–Terry 奖励损失

输入：
chosen_rewards: 浮点 Tensor [B] — 偏好回答的奖励。
rejected_rewards: 浮点 Tensor [B] — 非偏好回答的奖励。

返回（多项按元组顺序）：
loss: 浮点 Tensor [] — 零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch
import torch.nn.functional as F

def solve(chosen_rewards, rejected_rewards):
    return F.softplus(rejected_rewards - chosen_rewards).mean()

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(7, requires_grad=True), torch.randn(7, requires_grad=True))
    print(solve(*args))

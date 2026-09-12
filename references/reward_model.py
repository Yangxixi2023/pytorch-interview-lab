"""Bradley–Terry 奖励损失

输入：
chosen_rewards: 浮点 Tensor [B]
    偏好回答的奖励。
rejected_rewards: 浮点 Tensor [B]
    非偏好回答的奖励。

返回：
loss: 浮点 Tensor []
    零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch
import torch.nn.functional as F


def solve(chosen_rewards, rejected_rewards):
    # Bradley-Terry: P(chosen 更好) = sigmoid(r_chosen - r_rejected)。
    reward_margin = chosen_rewards - rejected_rewards
    sample_losses = F.softplus(-reward_margin)
    return sample_losses.mean()


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(7, requires_grad=True),
        torch.randn(7, requires_grad=True),
    )
    print(solve(*args))

"""GAE 广义优势估计

输入：
rewards: 浮点 Tensor [T, B]
    每个时间步的奖励；本题时间维在前。
values: 浮点 Tensor [T+1, B]
    状态价值，最后一行用于末端 bootstrap。
terminated: bool Tensor [T, B]
    True 表示该步后真正终止，阻断 bootstrap 与优势递推；不表示时间截断。
gamma: float —
    奖励折扣系数。
lam: float —
    GAE 衰减系数。

返回：
advantages: 浮点 Tensor [T, B]
    优势张量；必须 detach。
returns: 浮点 Tensor [T, B]
    advantages + values[:-1]；必须 detach。
多个返回值按上面顺序组成元组。
"""

import torch


def solve(rewards, values, terminated, gamma=0.99, lam=0.95):
    # 优势与回报是训练目标，不让梯度穿过价值估计。
    with torch.no_grad():
        time_steps, batch_size = rewards.shape
        advantages = torch.zeros_like(rewards)
        next_advantage = torch.zeros_like(rewards[0])  # [B]

        for time in range(time_steps - 1, -1, -1):
            # 真正终止时，bootstrap 与优势递推都需要切断。
            continues = (~terminated[time]).to(rewards.dtype)
            next_value = values[time + 1]
            td_error = rewards[time] + gamma * continues * next_value
            td_error = td_error - values[time]

            advantage = td_error + gamma * lam * continues * next_advantage
            advantages[time] = advantage
            next_advantage = advantage

        returns = advantages + values[:-1]
    return advantages, returns


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.ones(4, 2),
        torch.arange(10.0).reshape(5, 2).requires_grad_(),
        torch.tensor([[0, 0], [1, 0], [0, 0], [1, 0]], dtype=torch.bool),
    )
    print(solve(*args))

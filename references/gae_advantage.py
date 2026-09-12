"""GAE 广义优势估计

输入：
rewards: 浮点 Tensor [T, B] — 每个时间步的奖励；本题时间维在前。
values: 浮点 Tensor [T+1, B] — 状态价值，最后一行用于末端 bootstrap。
terminated: bool Tensor [T, B] — True 表示该步后真正终止，阻断 bootstrap 与优势递推；不表示时间截断。
gamma: float — — 奖励折扣系数。
lam: float — — GAE 衰减系数。

返回（多项按元组顺序）：
advantages: 浮点 Tensor [T, B] — 优势张量；必须 detach。
returns: 浮点 Tensor [T, B] — advantages + values[:-1]；必须 detach。
"""

import torch

def solve(rewards, values, terminated, gamma=0.99, lam=0.95):
    with torch.no_grad():
        a = torch.zeros_like(rewards)
        running = torch.zeros_like(rewards[0])
        for t in range(rewards.shape[0] - 1, -1, -1):
            live = (~terminated[t]).to(rewards.dtype)
            delta = rewards[t] + gamma * live * values[t + 1] - values[t]
            running = delta + gamma * lam * live * running
            a[t] = running
        return (a, a + values[:-1])

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.ones(4, 2), torch.arange(10.0).reshape(5, 2).requires_grad_(), torch.tensor([[0, 0], [1, 0], [0, 0], [1, 0]], dtype=torch.bool))
    print(solve(*args))

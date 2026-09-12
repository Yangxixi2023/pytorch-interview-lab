"""蒙特卡洛树搜索 PUCT 选择

输入：
priors: 浮点 Tensor [A]
    各候选动作的先验概率。
value_sums: 浮点 Tensor [A]
    各动作累计价值 W。
visits: 整数 Tensor [A]
    各动作访问次数 N，允许0。
parent_visits: int —
    父节点访问次数。
c_puct: float —
    探索项系数。

返回：
action_index: int —
    PUCT 最大动作的 Python 整数下标，并列取最小下标。
"""

import math
import torch


def solve(priors, value_sums, visits, parent_visits, c_puct=1.0):
    # 未访问的动作按题意 Q=0，不能直接除以访问次数0。
    mean_values = value_sums / visits.clamp_min(1)
    mean_values = torch.where(
        visits > 0, mean_values, torch.zeros_like(value_sums)
    )
    exploration_bonus = c_puct * priors * math.sqrt(parent_visits)
    exploration_bonus = exploration_bonus / (1 + visits)

    selection_scores = mean_values + exploration_bonus
    best_action = int(selection_scores.argmax())
    return best_action


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.tensor([0.2, 0.6, 0.2]),
        torch.tensor([5.0, 0.0, 1.0]),
        torch.tensor([10, 0, 2]),
        12,
        2.0,
    )
    print(solve(*args))

"""束搜索解码

输入：
log_probs: 浮点 Tensor [T, V]
    给定的每步 log probability；本题分布与历史无关。
beam_size: int —
    每步保留的最大候选数，正整数。

返回：
beams: list[tuple[list[int], float]] 最多 beam_size 项
    每项为 (token 序列, 累计 log 分数)；分数降序，并列时序列字典序升序。
"""

import torch


def solve(log_probs, beam_size):
    beams = [([], 0.0)]
    for step_log_probabilities in log_probs:
        candidates = []
        for sequence, accumulated_score in beams:
            for token, log_probability in enumerate(step_log_probabilities):
                next_sequence = sequence + [token]
                next_score = accumulated_score + float(log_probability)
                candidates.append((next_sequence, next_score))

        # 分数降序；并列时按 token 序列字典序升序。
        candidates.sort(key=lambda candidate: (-candidate[1], candidate[0]))
        beams = candidates[:beam_size]

    return beams


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.tensor([[-0.2, -1.0], [-1.2, -0.3], [-0.5, -0.8]]), 3)
    print(solve(*args))

"""实现 Softmax

输入：
x: 浮点 Tensor 任意形状
    输入 logits，可能包含很大的正数。
dim: int —
    进行归一化的维度，支持负索引。

返回：
probabilities: 浮点 Tensor 与 x 相同
    沿 dim 的和为 1，保留输入梯度；不能调用现成 softmax。
"""

import torch


def solve(x, dim=-1):
    # 减去最大值不会改变概率，却能避免 exp 溢出。
    maximum = x.amax(dim=dim, keepdim=True)
    shifted_logits = x - maximum
    unnormalized_probabilities = torch.exp(shifted_logits)
    normalizer = unnormalized_probabilities.sum(dim=dim, keepdim=True)
    probabilities = unnormalized_probabilities / normalizer
    return probabilities


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.tensor([[1000.0, 1001.0, 999.0]], requires_grad=True),)
    print(solve(*args))

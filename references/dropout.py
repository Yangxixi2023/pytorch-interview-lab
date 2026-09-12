"""实现 Dropout

输入：
x: 浮点 Tensor 任意形状
    待处理激活。
p: float —
    丢弃概率，0<=p<1。
training: bool —
    True 训练，False 推理。
keep_mask: bool Tensor 与 x 相同
    预先采样的保留掩码；本题不要求重新随机采样。

返回：
output: 浮点 Tensor 与 x 相同
    训练：x*keep_mask/(1-p)；推理：x。保留 x 梯度。
"""

import torch


def solve(x, p, training, keep_mask):
    if not training:
        return x
    # keep_mask 已由调用者提供；inverted dropout 在训练时补偿期望。
    retained_values = x * keep_mask
    output = retained_values / (1.0 - p)
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.ones(2, 4, requires_grad=True),
        0.5,
        True,
        torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool),
    )
    print(solve(*args))

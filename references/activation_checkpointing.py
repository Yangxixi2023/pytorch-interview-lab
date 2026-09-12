"""激活检查点

输入：
x: 浮点 Tensor [..., D]
    MLP 输入；可能不需要梯度。
w1: 浮点 Tensor [D, F]
    第一层可训练权重。
w2: 浮点 Tensor [F, O]
    第二层可训练权重。

返回：
output: 浮点 Tensor [..., O]
    checkpoint 包裹的 ReLU(x@w1)@w2，保留参数梯度。
"""

import torch


def solve(x, w1, w2):
    from torch.utils.checkpoint import checkpoint

    def mlp_forward(inputs, first_weight, second_weight):
        hidden_features = inputs @ first_weight
        activated_features = torch.relu(hidden_features)
        return activated_features @ second_weight

    # 非 reentrant 版本也支持输入无梯度、参数有梯度的情形。
    output = checkpoint(mlp_forward, x, w1, w2, use_reentrant=False)
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(2, 4, requires_grad=True),
        torch.randn(4, 6, requires_grad=True),
        torch.randn(6, 3, requires_grad=True),
    )
    print(solve(*args))

"""SFT Loss 因果语言模型

输入：
logits: 浮点 Tensor [B, T, V] — t 位置预测 t+1 标签，T>=2。
labels: int64 Tensor [B, T] — 目标 token ID，或 ignore_index；shift 后至少一个有效位置。
ignore_index: int — — 屏蔽 prompt/padding 的特殊标签值。

返回（多项按元组顺序）：
loss: 浮点 Tensor [] — 零维标量 Tensor；返回 loss 本身，不要调用 .item() 或转为 Python float。
"""

import torch

def solve(logits, labels, ignore_index=-100):
    lp = logits[:, :-1].log_softmax(-1)
    y = labels[:, 1:]
    valid = y != ignore_index
    return -lp[valid].gather(-1, y[valid, None]).mean()

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(2, 4, 5, requires_grad=True), torch.tensor([[-100, -100, 2, 1], [-100, 3, 0, -100]]))
    print(solve(*args))

"""K-means 聚类

输入：
x: 浮点 Tensor [N, D] — 待聚类数据。
initial_centers: 浮点 Tensor [K, D] — 指定的初始中心，不要随机初始化。
iterations: int — — 恰好执行的 Lloyd 更新次数，允许0。

返回（多项按元组顺序）：
centers: 浮点 Tensor [K, D] — 更新后的中心；空簇保留原中心。
labels: int64 Tensor [N] — 根据最终中心重新分配；并列最小索引。
"""

import torch

def solve(x, initial_centers, iterations):
    centers = initial_centers.clone()
    for _ in range(iterations):
        labels = (x[:, None] - centers[None, :]).square().sum(-1).argmin(-1)
        centers = torch.stack([x[labels == k].mean(0) if (labels == k).any() else centers[k] for k in range(len(centers))])
    labels = (x[:, None] - centers[None, :]).square().sum(-1).argmin(-1)
    return (centers, labels)

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.tensor([[0.0, 0.0], [1.0, 0.0], [9.0, 9.0], [10.0, 9.0]]), torch.tensor([[0.0, 0.0], [9.0, 9.0], [100.0, 100.0]]), 3)
    print(solve(*args))

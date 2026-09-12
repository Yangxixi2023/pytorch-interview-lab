"""K-means 聚类

输入：
x: 浮点 Tensor [N, D]
    待聚类数据。
initial_centers: 浮点 Tensor [K, D]
    指定的初始中心，不要随机初始化。
iterations: int —
    恰好执行的 Lloyd 更新次数，允许0。

返回：
centers: 浮点 Tensor [K, D]
    更新后的中心；空簇保留原中心。
labels: int64 Tensor [N]
    根据最终中心重新分配；并列最小索引。
多个返回值按上面顺序组成元组。
"""

import torch


def solve(x, initial_centers, iterations):
    centers = initial_centers.clone()
    cluster_count = centers.shape[0]

    for iteration in range(iterations):
        differences = x[:, None, :] - centers[None, :, :]
        squared_distances = differences.square().sum(dim=-1)  # [N,K]
        labels = squared_distances.argmin(dim=-1)

        updated_centers = []
        for cluster in range(cluster_count):
            members = x[labels == cluster]
            # 空簇保持原中心，这是题目约定的 Lloyd 更新规则。
            if members.shape[0] == 0:
                updated_centers.append(centers[cluster])
            else:
                updated_centers.append(members.mean(dim=0))
        centers = torch.stack(updated_centers)

    # 返回的 labels 必须对应最终中心，而非最后一次更新前的中心。
    differences = x[:, None, :] - centers[None, :, :]
    labels = differences.square().sum(dim=-1).argmin(dim=-1)
    return centers, labels


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [9.0, 9.0], [10.0, 9.0]]),
        torch.tensor([[0.0, 0.0], [9.0, 9.0], [100.0, 100.0]]),
        3,
    )
    print(solve(*args))

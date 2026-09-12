"""图读出（图级池化）

输入：
x: 浮点 Tensor [N, F]
    节点特征。
batch: int64 Tensor [N]
    每节点所属图的编号，范围 0..num_graphs-1。
num_graphs: int —
    图的总数量，可包含没有节点的图。

返回：
output: 浮点 Tensor [num_graphs, F]
    每图节点特征均值；空图输出全零。
"""

import torch


def solve(x, batch, num_graphs):
    feature_dim = x.shape[-1]
    graph_sums = x.new_zeros(num_graphs, feature_dim)
    graph_sums = graph_sums.index_add(dim=0, index=batch, source=x)
    node_counts = torch.bincount(batch, minlength=num_graphs)

    # 空图的和为0，除以1后仍为0，符合本题约定。
    graph_means = graph_sums / node_counts.clamp_min(1)[:, None]
    return graph_means


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(5, 3, requires_grad=True),
        torch.tensor([0, 2, 0, 2, 2]),
        4,
    )
    print(solve(*args))

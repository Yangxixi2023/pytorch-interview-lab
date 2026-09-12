"""MPNN 消息传递

输入：
x: 浮点 Tensor [N, F]
    每个节点的输入特征。
adj: 浮点 Tensor [N, N]
    adj[i,j] 表示 j 向 i 传递信息的边；输入无自环，GCN 要求对称邻接。
weight: 浮点 Tensor [F, O]
    特征变换矩阵。

返回：
output: 浮点 Tensor [N, O]
    按题目指定的消息与归约规则得到的节点特征；保留梯度。
"""

import torch


def solve(x, adj, weight):
    projected_features = x @ weight  # [N,O]
    received_features = adj @ projected_features
    weighted_degree = adj.sum(dim=-1, keepdim=True)

    # sum_j A[i,j] * (h_j-h_i)，按代数展开避免构造 [N,N,O]。
    output = received_features - weighted_degree * projected_features
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    a = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    args = (
        torch.randn(4, 3, requires_grad=True),
        a,
        torch.randn(3, 2, requires_grad=True),
    )
    print(solve(*args))

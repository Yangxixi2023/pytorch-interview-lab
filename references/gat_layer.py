"""GAT 层（图注意力）

输入：
x: 浮点 Tensor [N, F]
    节点输入特征。
adj: bool Tensor [N, N]
    可连接的邻接掩码，实现时再添加自环。
weight: 浮点 Tensor [F, O]
    特征投影。
attn_src: 浮点 Tensor [O]
    接收节点 i 的注意力向量。
attn_dst: 浮点 Tensor [O]
    发送节点 j 的注意力向量。

返回：
output: 浮点 Tensor [N, O]
    单头 GAT 聚合输出，不含末端激活。
"""

import torch
import torch.nn.functional as F


def solve(x, adj, weight, attn_src, attn_dst):
    node_features = x @ weight
    receiver_scores = node_features @ attn_src  # [N]
    sender_scores = node_features @ attn_dst  # [N]
    pair_scores = receiver_scores[:, None] + sender_scores[None, :]
    pair_scores = F.leaky_relu(pair_scores, negative_slope=0.2)

    node_count = adj.shape[0]
    self_loops = torch.eye(node_count, device=adj.device, dtype=torch.bool)
    allowed_edges = adj | self_loops
    masked_scores = pair_scores.masked_fill(~allowed_edges, float("-inf"))
    attention_weights = masked_scores.softmax(dim=-1)
    output = attention_weights @ node_features
    return output


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(4, 3, requires_grad=True),
        torch.zeros(4, 4, dtype=torch.bool),
        torch.randn(3, 2, requires_grad=True),
        torch.randn(2, requires_grad=True),
        torch.randn(2, requires_grad=True),
    )
    print(solve(*args))

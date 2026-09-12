"""链接预测

输入：
z: 浮点 Tensor [N, D]
    节点编码。
edges: int64 Tensor [2, E]
    第一行是源节点，第二行是目标节点。

返回：
probabilities: 浮点 Tensor [E]
    每条边的 sigmoid 内积概率，按输入边的顺序返回。
"""

import torch


def solve(z, edges):
    source_nodes = edges[0]
    target_nodes = edges[1]
    source_embeddings = z[source_nodes]
    target_embeddings = z[target_nodes]
    edge_logits = (source_embeddings * target_embeddings).sum(dim=-1)
    edge_probabilities = torch.sigmoid(edge_logits)
    return edge_probabilities


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (
        torch.randn(4, 3, requires_grad=True),
        torch.tensor([[0, 1, 0], [1, 2, 1]]),
    )
    print(solve(*args))

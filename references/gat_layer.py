"""GAT 层（图注意力）

输入：
x: 浮点 Tensor [N, F] — 节点输入特征。
adj: bool Tensor [N, N] — 可连接的邻接掩码，实现时再添加自环。
weight: 浮点 Tensor [F, O] — 特征投影。
attn_src: 浮点 Tensor [O] — 接收节点 i 的注意力向量。
attn_dst: 浮点 Tensor [O] — 发送节点 j 的注意力向量。

返回（多项按元组顺序）：
output: 浮点 Tensor [N, O] — 单头 GAT 聚合输出，不含末端激活。
"""

import torch
import torch.nn.functional as F

def solve(x, adj, weight, attn_src, attn_dst):
    h = x @ weight
    e = F.leaky_relu((h @ attn_src)[:, None] + (h @ attn_dst)[None, :], 0.2)
    mask = adj | torch.eye(len(adj), device=adj.device, dtype=torch.bool)
    return e.masked_fill(~mask, float('-inf')).softmax(-1) @ h

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(4, 3, requires_grad=True), torch.zeros(4, 4, dtype=torch.bool), torch.randn(3, 2, requires_grad=True), torch.randn(2, requires_grad=True), torch.randn(2, requires_grad=True))
    print(solve(*args))

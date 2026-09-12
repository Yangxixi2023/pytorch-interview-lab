"""图读出（图级池化）

输入：
x: 浮点 Tensor [N, F] — 节点特征。
batch: int64 Tensor [N] — 每节点所属图的编号，范围 0..num_graphs-1。
num_graphs: int — — 图的总数量，可包含没有节点的图。

返回（多项按元组顺序）：
output: 浮点 Tensor [num_graphs, F] — 每图节点特征均值；空图输出全零。
"""

import torch

def solve(x, batch, num_graphs):
    out = x.new_zeros(num_graphs, x.shape[-1]).index_add(0, batch, x)
    count = torch.bincount(batch, minlength=num_graphs).clamp_min(1)
    return out / count[:, None]

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (torch.randn(5, 3, requires_grad=True), torch.tensor([0, 2, 0, 2, 2]), 4)
    print(solve(*args))

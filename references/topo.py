"""拓扑排序

输入：
num_nodes: int —
    节点编号 0..num_nodes-1，允许0。
edges: list[tuple[int, int]] 任意长度
    (u,v) 表示 u→v，无重复边。

返回：
order: list[int] num_nodes 项或 []
    字典序最小的拓扑序，含孤立节点；存在环返回 []。
"""

import torch


def solve(num_nodes, edges):
    import heapq

    neighbors = [[] for _ in range(num_nodes)]
    indegree = [0] * num_nodes
    for source, target in edges:
        neighbors[source].append(target)
        indegree[target] += 1

    # 用最小堆，让每一步都选当前可选节点中编号最小的。
    ready = []
    for node in range(num_nodes):
        if indegree[node] == 0:
            heapq.heappush(ready, node)

    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for target in neighbors[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)

    if len(order) != num_nodes:
        return []  # 有环时，无法移除全部节点。
    return order


if __name__ == "__main__":
    torch.manual_seed(17)
    args = (6, [(0, 2), (1, 2), (2, 3), (1, 4)])
    print(solve(*args))

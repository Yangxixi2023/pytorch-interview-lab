"""拓扑排序

输入：
num_nodes: int — — 节点编号 0..num_nodes-1，允许0。
edges: list[tuple[int, int]] 任意长度 — (u,v) 表示 u→v，无重复边。

返回（多项按元组顺序）：
order: list[int] num_nodes 项或 [] — 字典序最小的拓扑序，含孤立节点；存在环返回 []。
"""

import torch

def solve(num_nodes, edges):
    import heapq
    adj = [[] for _ in range(num_nodes)]
    degree = [0] * num_nodes
    for u, v in edges:
        adj[u].append(v)
        degree[v] += 1
    heap = [i for i in range(num_nodes) if degree[i] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        u = heapq.heappop(heap)
        out.append(u)
        for v in adj[u]:
            degree[v] -= 1
            if degree[v] == 0:
                heapq.heappush(heap, v)
    return out if len(out) == num_nodes else []

if __name__ == "__main__":
    torch.manual_seed(17)
    args = (6, [(0, 2), (1, 2), (2, 3), (1, 4)])
    print(solve(*args))

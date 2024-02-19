import numpy as np
import torch


def vectorise_edge_index(adj: np.ndarray):
    n = adj.shape[0]
    edges = [(x, y) for x in range(n) for y in range(n) if adj[x][y] != 0]
    mp = {edge: idx for idx, edge in enumerate(edges)}
    reverse_idx = torch.tensor([mp[(y, x)] for x, y in edges], dtype=torch.int64)
    return torch.tensor(edges).T.contiguous(), reverse_idx


class ProblemInstance:
    def __init__(self, adj: np.ndarray, start: int, edge_weight: np.ndarray, random_numbers: np.ndarray):
        self.adj = np.copy(adj)
        self.start = start
        self.edge_weight = edge_weight
        n = adj.shape[0]
        edge_index, reverse_idx = vectorise_edge_index(self.adj + np.eye(n))
        self.edge_index_with_self_loops = edge_index
        self.reverse_idx = reverse_idx
        self.out_nodes = [[] for _ in range(n)]
        for x, y in filter(lambda edge: edge[0] != edge[1], edge_index.T):
            self.out_nodes[x].append(y)
        self.random_numbers = random_numbers

    def vectorise(self, position_dim: int):
        n = self.adj.shape[0]
        node_fts = torch.zeros(n, dtype=torch.int64)

        if self.start is not None:
            node_fts[self.start] = 1

        edge_fts = torch.zeros(self.edge_index_with_self_loops.shape[1], dtype=torch.int64)
        if self.start is not None:
            self_loop_mask = (self.edge_index_with_self_loops[0] == self.edge_index_with_self_loops[1])
            start_self_loop_mask = torch.logical_and(self_loop_mask, self.edge_index_with_self_loops[0] == self.start)
            edge_fts[start_self_loop_mask] = 1

        scalars = None
        position = (1.0 * torch.arange(n)) / n
        if position_dim >= 0:
            scalars = position[self.edge_index_with_self_loops[position_dim]]
        if self.edge_weight is not None:
            weights = self.edge_weight[self.edge_index_with_self_loops[0], self.edge_index_with_self_loops[1]]
            weights = torch.tensor(weights, dtype=torch.float64)
            scalars = weights

        if self.random_numbers is not None:
            assert self.edge_weight is None  # support only weights or random values
            scalars = torch.tensor(self.random_numbers, dtype=torch.float64)  # (num_steps, N)
            scalars = scalars[:, self.edge_index_with_self_loops[0]]
            scalars = torch.transpose(scalars, 1, 0)  # (E, num_steps)

        return (node_fts, self.edge_index_with_self_loops, self.reverse_idx, edge_fts, scalars)

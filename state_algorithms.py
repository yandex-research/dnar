import numpy as np

import abstracts

BFS_NODE_STATES, BFS_EDGE_STATES = range(2), range(2)
BFS_NOT_DISCOVERED, BFS_DISCOVERED = BFS_NODE_STATES
BFS_EDGE_NONE, BFS_EDGE_POINTER = BFS_EDGE_STATES

DFS_NODE_STATES, DFS_EDGE_STATES = range(5), range(2)
DFS_NOT_IN_THE_STACK, DFS_ON_TOP_OF_THE_STACK, DFS_ON_THE_STACK, DFS_PREEND, DFS_END = DFS_NODE_STATES
DFS_EDGE_NONE, DFS_EDGE_POINTER = DFS_EDGE_STATES

DIJKSTRA_NODE_STATES, DIJKSTRA_EDGE_STATES = range(2), range(3)
DIJKSTRA_NOT_IN_THE_TREE, DIJKSTRA_IN_THE_TREE = DIJKSTRA_NODE_STATES
DIJKSTRA_EDGE_NONE, DIJKSTRA_EDGE_POINTER, DIJKSTRA_EDGE_PREPOINTER = DIJKSTRA_EDGE_STATES

PRIM_NODE_STATES, PRIM_EDGE_STATES = range(2), range(2)
PRIM_NOT_IN_THE_TREE, PRIM_IN_THE_TREE = PRIM_NODE_STATES
PRIM_EDGE_NONE, PRIM_EDGE_POINTER = PRIM_EDGE_STATES

MIS_NODE_STATES, MIS_EDGE_STATES = range(4), range(1)
MIS_ALIVE, MIS_IN_MIS, MIS_DEAD_NOT_IN_MIS, MIS_WAIT = MIS_NODE_STATES
MIS_EDGE_NONE = MIS_EDGE_STATES


def flat(states, adj):
    n = adj.shape[0]
    return np.array([states[x][y] for x in range(n) for y in range(n) if adj[x][y]])


def new_states(node_states, edge_states):
    node_states.append(np.copy(node_states[-1]))
    edge_states.append(np.copy(edge_states[-1]))


def bfs_states(instance: abstracts.ProblemInstance):
    n = instance.adj.shape[0]
    adj = instance.adj + np.eye(n)

    node_states = [np.array([BFS_NOT_DISCOVERED] * n)]
    node_states[0][instance.start] = BFS_DISCOVERED

    edge_states = [np.ones((n, n), dtype=np.int32) * BFS_EDGE_NONE]
    edge_states[0][instance.start][instance.start] = BFS_EDGE_POINTER

    for i in range(n):
        new_states(node_states, edge_states)

        for node in range(n):
            if node_states[-2][node] != BFS_DISCOVERED:
                continue
            for out in instance.out_nodes[node]:
                if node_states[-1][out] == BFS_NOT_DISCOVERED:
                    node_states[-1][out] = BFS_DISCOVERED
                    edge_states[-1][out][node] = BFS_EDGE_POINTER

    return np.array(node_states)[1:], np.array([flat(state, adj) for state in edge_states[1:]])


def dfs_states(instance: abstracts.ProblemInstance):
    n = instance.adj.shape[0]
    adj = instance.adj + np.eye(n)

    node_states = [np.array([DFS_NOT_IN_THE_STACK] * n)]
    node_states[0][instance.start] = DFS_ON_TOP_OF_THE_STACK
    edge_states = [np.ones((n, n), dtype=np.int32) * DFS_EDGE_NONE]
    edge_states[0][instance.start][instance.start] = DFS_EDGE_POINTER

    def rec_dfs(current_node, prev=-1):
        nonlocal n
        assert node_states[-1][current_node] == DFS_ON_TOP_OF_THE_STACK
        for out in instance.out_nodes[current_node]:
            if node_states[-1][out] == DFS_NOT_IN_THE_STACK:
                new_states(node_states, edge_states)
                node_states[-1][current_node] = DFS_ON_THE_STACK
                node_states[-1][out] = DFS_ON_TOP_OF_THE_STACK
                edge_states[-1][out][current_node] = DFS_EDGE_POINTER
                rec_dfs(out, current_node)
                new_states(node_states, edge_states)
                node_states[-1][current_node] = DFS_ON_TOP_OF_THE_STACK
                node_states[-1][out] = DFS_END
        new_states(node_states, edge_states)
        node_states[-1][current_node] = DFS_PREEND
        if prev == -1:
            new_states(node_states, edge_states)
            node_states[-1][current_node] = DFS_END

    rec_dfs(instance.start)
    # pad the trajectory size
    while len(node_states) <= 3 * n:
        new_states(node_states, edge_states)
    assert len(node_states) == 3 * n + 1

    return np.array(node_states)[1:], np.array([flat(state, adj) for state in edge_states[1:]])


def prim_states(instance: abstracts.ProblemInstance):
    n = instance.adj.shape[0]
    adj = instance.adj + np.eye(n)

    tree_nodes = [instance.start]

    node_states = [np.array([PRIM_NOT_IN_THE_TREE] * n)]
    node_states[0][instance.start] = PRIM_IN_THE_TREE
    edge_states = [np.ones((n, n), dtype=np.int32) * PRIM_EDGE_NONE]
    edge_states[0][instance.start][instance.start] = PRIM_EDGE_POINTER

    for _ in range(1, n):
        best_weight = 1e9
        best_edge = None
        for mst_node in tree_nodes:
            assert node_states[-1][mst_node] == PRIM_IN_THE_TREE
            for new_node in instance.out_nodes[mst_node]:
                if node_states[-1][new_node] != PRIM_NOT_IN_THE_TREE:
                    continue
                if instance.edge_weight[mst_node][new_node] < best_weight:
                    best_weight = instance.edge_weight[mst_node][new_node]
                    best_edge = (mst_node, new_node)
        assert best_edge is not None

        mst_node, new_node = best_edge

        new_states(node_states, edge_states)
        node_states[-1][new_node] = PRIM_IN_THE_TREE
        edge_states[-1][new_node][mst_node] = PRIM_EDGE_POINTER

        tree_nodes.append(new_node)
    new_states(node_states, edge_states)

    return np.array(node_states)[1:], np.array([flat(state, adj) for state in edge_states[1:]])


def mis_states(instance: abstracts.ProblemInstance):
    n = instance.adj.shape[0]
    adj = instance.adj

    alive = np.ones(n, dtype=bool)
    in_mis = np.zeros(n, dtype=bool)

    node_states = [np.array([MIS_ALIVE] * n, dtype=np.int32)]
    edge_states = [np.ones((n, n), dtype=np.int32) * MIS_EDGE_NONE]

    step = 0

    while np.any(alive):
        new_states(node_states, edge_states)
        random_numbers = instance.random_numbers[step]
        step += 2
        for node in filter(lambda x: alive[x], range(n)):
            if random_numbers[node] < random_numbers[np.logical_and(adj[node], alive)].min(initial=1.):
                in_mis[node] = True
                node_states[-1][node] = MIS_IN_MIS
            else:
                node_states[-1][node] = MIS_WAIT

        new_states(node_states, edge_states)
        new_alive = np.copy(alive)
        for node in filter(lambda x: alive[x], range(n)):
            if in_mis[node]:
                new_alive[node] = False
            elif np.any(in_mis[adj[node].astype(bool)]):
                new_alive[node] = False
                node_states[-1][node] = MIS_DEAD_NOT_IN_MIS
            else:
                node_states[-1][node] = MIS_ALIVE
        alive = new_alive

    while len(node_states) <= n:
        new_states(node_states, edge_states)

    return np.array(node_states)[1:], np.array([flat(state, adj + np.eye(n)) for state in edge_states[1:]])


def dijkstra_states(instance: abstracts.ProblemInstance):
    n = instance.adj.shape[0]
    adj = instance.adj + np.eye(n)
    tree_nodes = [instance.start]

    node_states = [np.array([DIJKSTRA_NOT_IN_THE_TREE] * n)]
    node_states[0][instance.start] = DIJKSTRA_IN_THE_TREE
    edge_states = [np.ones((n, n), dtype=np.int32) * DIJKSTRA_EDGE_NONE]
    edge_states[0][instance.start][instance.start] = DIJKSTRA_EDGE_POINTER
    edge_weight = np.copy(instance.edge_weight)

    INF = 1e9
    distance = np.ones(n, dtype=np.float64)
    distance[instance.start] = 0.

    last_prepointer = None

    for _ in range(1, n):
        best_weight = INF
        best_edge = None
        for mst_node in tree_nodes:
            assert node_states[-1][mst_node] == DIJKSTRA_IN_THE_TREE
            for new_node in instance.out_nodes[mst_node]:
                if node_states[-1][new_node] != DIJKSTRA_NOT_IN_THE_TREE:
                    continue
                if edge_weight[mst_node][new_node] < best_weight:
                    best_weight = edge_weight[mst_node][new_node]
                    best_edge = (mst_node, new_node)
        assert best_edge is not None

        mst_node, new_node = best_edge

        new_states(node_states, edge_states)
        if last_prepointer is not None:
            edge_states[-1][last_prepointer[0]][last_prepointer[1]] = DIJKSTRA_EDGE_POINTER
        node_states[-1][new_node] = DIJKSTRA_IN_THE_TREE
        edge_states[-1][new_node][mst_node] = DIJKSTRA_EDGE_PREPOINTER
        last_prepointer = (new_node, mst_node)
        for out in instance.out_nodes[new_node]:
            edge_weight[new_node][out] += edge_weight[mst_node][new_node]

        tree_nodes.append(new_node)
    new_states(node_states, edge_states)  # for exactly n steps
    assert last_prepointer is not None
    edge_states[-1][last_prepointer[0]][last_prepointer[1]] = DIJKSTRA_EDGE_POINTER

    return np.array(node_states)[1:], np.array([flat(state, adj) for state in edge_states[1:]])


def dijkstra_scalars_update(edge_fts, scalars, edge_index, batch_reverse_idx):
    from torch_geometric.utils import scatter
    prepointer_edge = edge_fts[:, DIJKSTRA_EDGE_PREPOINTER].detach()
    forward = prepointer_edge[batch_reverse_idx]
    node_accum = scatter(forward * scalars, index=edge_index[1], reduce='sum')
    return scalars + node_accum[edge_index[0]]

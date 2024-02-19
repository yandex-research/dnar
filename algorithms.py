from collections import deque

import numpy as np

import abstracts


def bfs(instance: abstracts.ProblemInstance):
    n = instance.adj.shape[0]
    nodes = deque([instance.start])
    distance = -1 * np.ones(n)
    distance[instance.start] = 0
    pred = -np.ones(n, dtype=np.int32)
    pred[instance.start] = instance.start

    while nodes:
        cur = nodes.popleft()
        for out in instance.out_nodes[cur]:
            if distance[out] == distance[cur] + 1:
                pred[out] = min(pred[out], cur)
            if pred[out] != -1:
                continue
            pred[out] = cur
            distance[out] = 1 + distance[cur]
            nodes.append(out)
    return pred


def dfs(instance: abstracts.ProblemInstance):
    n = instance.adj.shape[0]
    pred = -np.ones(n, dtype=np.int32)
    pred[instance.start] = instance.start

    def rec_dfs(cur):
        nonlocal n
        for out in instance.out_nodes[cur]:
            if pred[out] != -1:
                continue
            pred[out] = cur
            rec_dfs(out)

    rec_dfs(instance.start)
    return pred


def prim(instance: abstracts.ProblemInstance):
    assert instance.edge_weight is not None
    n = instance.adj.shape[0]
    pred = -np.ones(n, dtype=np.int32)
    pred[instance.start] = instance.start
    tree_nodes = [instance.start]

    for _ in range(1, n):
        best_weight = 1e9
        best_edge = None
        for mst_node in tree_nodes:
            assert pred[mst_node] != -1
            for new_node in instance.out_nodes[mst_node]:
                if pred[new_node] != -1:
                    continue
                if instance.edge_weight[mst_node][new_node] < best_weight:
                    best_weight = instance.edge_weight[mst_node][new_node]
                    best_edge = (mst_node, new_node)
        assert best_edge is not None
        mst_node, new_node = best_edge
        pred[new_node] = mst_node
        tree_nodes.append(new_node)

    return pred


def mis(instance: abstracts.ProblemInstance):
    adj = instance.adj
    n = adj.shape[0]
    alive = np.ones(n, dtype=bool)
    in_mis = np.zeros(n, dtype=bool)
    step = 0

    while np.any(alive):
        random_numbers = instance.random_numbers[step]
        step += 2
        for node in filter(lambda x: alive[x], range(n)):
            in_mis[node] = random_numbers[node] < random_numbers[np.logical_and(
                adj[node], alive)].min(initial=1.)
        new_alive = np.copy(alive)
        for node in filter(lambda x: alive[x], range(n)):
            if in_mis[node] or np.any(in_mis[adj[node].astype(bool)]):
                new_alive[node] = False
        alive = new_alive

    return in_mis


def dijkstra(instance: abstracts.ProblemInstance):
    assert instance.edge_weight is not None
    n = instance.adj.shape[0]
    pred = -np.ones(n, dtype=np.int32)
    pred[instance.start] = instance.start

    INF = 1e9
    distance = np.ones(n, dtype=np.float64) * INF
    visited = np.zeros(n, dtype=bool)
    distance[instance.start] = 0.

    for _ in range(1, n):
        best_distance = INF
        best_node = -1
        for node in range(n):
            if visited[node]:
                continue
            if distance[node] < best_distance:
                best_distance = distance[node]
                best_node = node
        assert best_node != -1
        visited[best_node] = True
        for out in instance.out_nodes[best_node]:
            if distance[out] > distance[best_node] + instance.edge_weight[best_node][out]:
                distance[out] = distance[best_node] + instance.edge_weight[best_node][out]
                pred[out] = best_node
    return pred


def test_consistency(config_path: str):
    import tqdm

    import pipelines
    from configs import base_config
    from data_utils import ErdosRenyiGraphSampler

    config = base_config.read_config(config_path)
    pipeline = pipelines.PIPELINES[config.pipeline_name]
    assert config.output_location in ('edge', 'node')

    num_it = 1000
    num_nodes = 16

    sampler = ErdosRenyiGraphSampler(config)
    for _ in tqdm.tqdm(range(num_it), config_path):
        instance = sampler(num_nodes)
        algo_output = pipeline.algorithm(instance)
        states_algo_output = pipeline.states_algorithm(instance)
        if config.output_location == 'edge':  # pointer
            edge_index = instance.edge_index_with_self_loops.numpy()
            algo_pointers = edge_index[1] == algo_output[edge_index[0]]
            states_pointers = states_algo_output[1][-1] == 1
            assert np.all(algo_pointers == states_pointers)
        if config.output_location == 'node':  # node mask
            states_masks = states_algo_output[0][-1] == 1
            assert np.all(algo_output == states_masks)


if __name__ == "__main__":
    test_consistency("./configs/bfs_stepwise.yaml")
    test_consistency("./configs/dfs_stepwise.yaml")
    test_consistency("./configs/dijkstra_stepwise.yaml")
    test_consistency("./configs/prim_stepwise.yaml")
    test_consistency("./configs/mis_stepwise.yaml")

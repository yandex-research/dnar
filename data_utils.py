import math

import numpy as np
import torch
import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import algorithms
import pipelines
from abstracts import ProblemInstance
from configs import base_config


def er_probabilities(n):
    base = math.log(n) / n
    return (base * 1, base * 3)


class ErdosRenyiGraphSampler:
    def __init__(self, config: base_config.Config):
        self.connected = True
        self.weighted = config.weighted_graphs
        self.generate_random_numbers = config.generate_random_numbers
        self.generate_start = config.generate_start

    def __call__(self, num_nodes):
        p_segment = er_probabilities(num_nodes)
        p = np.random.uniform(p_segment[0], p_segment[1])

        w = None
        random_numbers = None
        start = None

        if self.generate_start:
            start = np.random.randint(0, num_nodes - 1)
        while True:
            adj = np.triu(np.random.binomial(1, p, size=(num_nodes, num_nodes)), k=1)
            adj += adj.T
            if self.weighted:
                w = np.triu(np.random.uniform(0., 1., (num_nodes, num_nodes)), k=1)
                w *= adj
                w = (w + w.T)
            if self.generate_random_numbers:
                random_numbers = np.random.rand(num_nodes, num_nodes)  # steps count bounded by num_nodes
                random_numbers[np.arange(1, num_nodes, 2)] = np.ones(num_nodes)  # for mis, need scalars on even steps
            instance = ProblemInstance(adj, start, w, random_numbers)
            instance.start = 0
            if self.connected and algorithms.bfs(instance).min() < 0:
                continue
            instance.start = start
            return instance


GRAPH_SAMPLERS = {
    'er': ErdosRenyiGraphSampler,
}


class DataGenerator:
    def __init__(self, config: base_config.Config, pipeline: pipelines.Pipeline, split: str):
        self.split = split
        self.num_nodes = config.problem_size[split]
        self.samples_count = config.samples_count[split]
        self.create_hints = split == 'train' and config.use_hints

        self.batch_size = config.batch_size
        sampler_class = GRAPH_SAMPLERS[config.sampler_name[split]]
        self.sampler = sampler_class(config)
        self.position_dim = config.position_dim

        self.algorithm = pipeline.algorithm
        self.states_algorithm = pipeline.states_algorithm

        self.node_states_count = config.node_states_count
        self.edge_states_count = config.edge_states_count

    def as_dataloader(self, shuffle: bool = False):
        datapoints = []

        for _ in tqdm.tqdm(range(self.samples_count), 'Generate samples for {}'.format(self.split)):
            instance = self.sampler(self.num_nodes)
            y = torch.tensor(self.algorithm(instance))
            mid_y, mid_edge_y = None, None
            if self.create_hints:
                node_states, edge_states = self.states_algorithm(instance)
                mid_y = torch.transpose(torch.tensor(node_states, dtype=torch.int64), 1, 0)
                mid_edge_y = torch.transpose(torch.tensor(edge_states, dtype=torch.int64), 1, 0)

            node_fts, edge_index, reverse_idx, edge_fts, scalars = instance.vectorise(position_dim=self.position_dim)
            node_fts = 1. * torch.nn.functional.one_hot(node_fts, self.node_states_count)
            edge_fts = 1. * torch.nn.functional.one_hot(edge_fts, self.edge_states_count)

            datapoints.append(Data(node_fts=node_fts, edge_attr=edge_fts,
                                   edge_index=edge_index, reverse_idx=reverse_idx, y=y, mid_y=mid_y,
                                   mid_edge_y=mid_edge_y, num_nodes=self.num_nodes, scalars=scalars))

        return DataLoader(datapoints, batch_size=self.batch_size, shuffle=shuffle)

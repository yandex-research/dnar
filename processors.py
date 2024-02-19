import math

import numpy as np
import torch
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import LayerNorm, MessagePassing
from torch_geometric.utils import group_argsort, softmax

from configs import base_config


# mirror pytorch implementation, but with softmax_index, needed for batched graphs
def gumbel_softmax(logits, softmax_index, tau, use_noise):
    if use_noise:
        noise = (-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log())
        logits = logits + noise
    if tau == 0.:
        # no detach trick as we do not use hard at training time
        if softmax_index is None:
            index = logits.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        else:
            y_hard = 1. * (group_argsort(logits, softmax_index, descending=True) == 0)
        return y_hard
    logits = logits / tau
    if softmax_index is None:
        return logits.softmax(-1)
    return softmax(logits, softmax_index)


class Transformer(MessagePassing):
    def __init__(self, config: base_config.Config):
        super().__init__(aggr='add')
        h = config.h
        self.h = h
        self.num_iterations = config.num_iterations
        self.message_top_one_choice = config.message_top_one_choice
        self.temp_by_step = np.geomspace(config.processor_upper_t, config.processor_lower_t, config.num_iterations + 2)
        self.temp_on_eval = config.temp_on_eval

        self.lin_query = Linear(h, h, bias=False)
        self.lin_key = Linear(h, h, bias=False)
        self.lin_value = Linear(h, h, bias=False)

        self.edge_key = Linear(h, h, bias=False)

        self.node_mlp = Sequential(Linear(3 * h, h, bias=False), ReLU())

        self.node_to_edge = Sequential(Linear(2 * h, h, bias=False), ReLU())
        self.edge_mlp = Sequential(Linear(4 * h, h, bias=False), ReLU(), Linear(h, h, bias=False), ReLU())

        self.ln_nodes = LayerNorm(h, mode='node')
        self.ln_edges = LayerNorm(h, mode='node')

        self._alpha = None
        self.scalar_w = torch.nn.Parameter(-torch.ones(1))
        self.use_noise = config.use_noise

    def forward(self, node_fts, edge_fts, edge_index, reverse_idx, scalars, batch, training_step):
        Q = self.lin_query(node_fts)
        K = self.lin_key(node_fts)
        V = self.lin_value(node_fts)

        edge_K = self.edge_key(edge_fts)

        self._alpha = None
        uns_scalars = torch.unsqueeze(scalars, -1) * self.scalar_w

        out_nodes = self.propagate(Q=Q, K=K, V=V, edge_K=edge_K, edge_index=edge_index,
                                   batch=batch, scalars=uns_scalars, training_step=training_step)
        idxs = edge_index[0, edge_index[0] == edge_index[1]]
        assert torch.all(torch.arange(node_fts.shape[0]).to(node_fts.device) == idxs), idxs
        self_loop_alpha = torch.ones_like(node_fts) * torch.unsqueeze(self._alpha[[edge_index[0] == edge_index[1]]], -1)
        out_nodes = self.node_mlp(torch.hstack([out_nodes, node_fts, self_loop_alpha]))

        node_to_edge = self.node_to_edge(torch.hstack([K[edge_index[0]], Q[edge_index[1]]]))

        alpha = torch.ones_like(edge_fts) * torch.unsqueeze(self._alpha, -1)
        out_edges = self.edge_mlp(torch.hstack([edge_fts, node_to_edge, alpha[reverse_idx], alpha]))

        out_nodes = self.ln_nodes(out_nodes)
        out_edges = self.ln_edges(out_edges)

        return out_nodes, out_edges

    def message(self, Q_i, K_j, V_j, edge_K, edge_index, scalars, batch, training_step):
        alpha = (Q_i * (K_j + edge_K) + scalars).sum(dim=-1) / math.sqrt(self.h)

        softmax_index = batch[edge_index[0]] if self.message_top_one_choice == 'global' else edge_index[1]

        tau = self.temp_on_eval if training_step == -1 else self.temp_by_step[training_step]
        use_noise = self.use_noise and training_step != -1
        alpha = gumbel_softmax(alpha, softmax_index, tau=tau, use_noise=use_noise)
        self._alpha = alpha

        return V_j * alpha.view(-1, 1)

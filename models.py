import numpy as np
import torch
from torch.nn import Linear, Sequential
from torch.nn.functional import cross_entropy
from torch_geometric.utils import scatter

import pipelines
from configs import base_config
from processors import Transformer, gumbel_softmax


def batch_reverse_index(edge_index, batch, reverse_idx):
    edge_batch = batch[edge_index[0]]
    edge_count = scatter(torch.ones_like(edge_index[0]), index=edge_batch, reduce='sum')
    edge_count = torch.cumsum(edge_count, dim=0)
    batch_edge_count = edge_count[edge_batch - 1]
    batch_edge_count[edge_batch == 0] = 0
    return reverse_idx + batch_edge_count


class NarModel(torch.nn.Module):
    def __init__(self, config: base_config.Config):
        super().__init__()
        h = config.h
        self.h = h

        self.processor = Transformer(config)

        self.stepwise_training = config.stepwise_training
        self.steps_counter = lambda num_nodes: int(num_nodes * config.steps_mul)

        self.node_states_count = config.node_states_count
        self.edge_states_count = config.edge_states_count

        self.node_encoder = Linear(config.node_states_count, h, bias=False)
        self.edge_encoder = Linear(config.edge_states_count, h, bias=False)

        self.node_projection = Linear(h, config.node_states_count, bias=False)
        self.edge_projection = Linear(h, config.edge_states_count, bias=False)

        self.use_discrete_bottleneck = config.use_discrete_bottleneck
        self.temp_by_step = np.geomspace(config.states_upper_t, config.states_lower_t, config.num_iterations + 2)
        self.temp_on_eval = config.temp_on_eval
        self.use_noise = config.use_noise
        train_steps = self.steps_counter(config.problem_size['train'])
        self.loss_weight = np.geomspace(train_steps, 1e-3, train_steps) if config.hint_loss_weight else None

        self.output_from_nodes = config.output_location == 'node'
        states_count = config.node_states_count if self.output_from_nodes else config.edge_states_count
        # decoder is redundant, as we can use last states as predictions, keep for consistency
        self.decoder = Sequential(Linear(states_count, h), Linear(h, 1))
        self.scalars_update = pipelines.PIPELINES[config.pipeline_name].scalars_update

    def forward(self, node_fts, edge_fts, edge_index, reverse_idx, scalars, batch, training_step: int = -1, hints=None):
        training_time = training_step != -1
        processor_steps_count = self.steps_counter((batch == 0).sum().item())

        batch_reverse_idx = batch_reverse_index(edge_index, batch, reverse_idx)

        hint_loss = torch.tensor(0.).to(node_fts.device)

        for processor_step in range(processor_steps_count):
            if self.scalars_update is not None:
                scalars = self.scalars_update(edge_fts, scalars, edge_index, batch_reverse_idx)

            current_step_scalars = scalars[:, processor_step] if len(scalars.shape) > 1 else scalars

            if not training_time:
                current_step_scalars = current_step_scalars.detach().clone() / current_step_scalars.max()

            node_fts = self.node_encoder(node_fts)
            edge_fts = self.edge_encoder(edge_fts)

            node_fts, edge_fts = self.processor(
                node_fts, edge_fts, edge_index, batch_reverse_idx,
                current_step_scalars, batch, training_step=training_step)

            node_fts = self.node_projection(node_fts)  # (N, K)
            edge_fts = self.edge_projection(edge_fts)

            if hints is not None:
                node_states = hints[0][:, processor_step]
                edge_states = hints[1][:, processor_step]
                hint_loss += self.compute_states_loss(
                    node_fts, edge_fts, node_states, edge_states, processor_step) / processor_steps_count

            if self.use_discrete_bottleneck:
                tau = self.temp_by_step[training_step] if training_time else self.temp_on_eval
                use_noise = self.use_noise and training_time
                node_fts = gumbel_softmax(node_fts, softmax_index=None, tau=tau, use_noise=use_noise)
                edge_fts = gumbel_softmax(edge_fts, softmax_index=None, tau=tau, use_noise=use_noise)

            if self.stepwise_training and training_time:
                assert hints is not None
                node_fts, edge_fts = self.teacher_force_state(hints[0][:, processor_step], hints[1][:, processor_step])

        fts = node_fts if self.output_from_nodes else edge_fts
        preds = self.decoder(fts).squeeze()
        if hints is not None:
            return (preds, hint_loss)
        return preds

    def compute_states_loss(self, node_fts, edge_fts, node_states, edge_states, processor_step):
        current_step_states_loss = cross_entropy(node_fts, node_states) + cross_entropy(edge_fts, edge_states)
        if self.loss_weight is not None:
            current_step_states_loss *= self.loss_weight[processor_step]
        return current_step_states_loss

    def teacher_force_state(self, node_states, edge_states):
        node_fts = 1. * torch.nn.functional.one_hot(node_states, self.node_states_count)
        edge_fts = 1. * torch.nn.functional.one_hot(edge_states, self.edge_states_count)
        return node_fts, edge_fts

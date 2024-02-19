from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.utils import group_argsort, softmax


def node_pointer_loss(gt, prediction, edge_index, batch):
    combined_gt = gt + batch * (batch == 0).sum()
    gt_mask = 1. * (edge_index[1] == combined_gt[edge_index[0]])

    prediction_probs = softmax(prediction, edge_index[0])
    return -torch.mean(gt_mask * torch.log(prediction_probs + 1e-12))


def node_mask_loss(gt, prediction, edge_index, batch):
    return torch.nn.functional.binary_cross_entropy_with_logits(prediction, gt.float())


def node_pointer_score(graph, prediction):
    edge_index = graph.edge_index
    is_gt_pointer = edge_index[1] == graph.y[edge_index[0]]
    is_predicted_pointer = group_argsort(prediction, edge_index[0], descending=True) == 0
    return torch.logical_and(is_gt_pointer, is_predicted_pointer).sum() / graph.num_nodes


def node_pointer_all_match(graph, prediction):
    return 1.0 * (node_pointer_score(graph, prediction) == 1.)


def node_mask_accuracy_node_level(graph, prediction):
    pred_mask = torch.nn.functional.sigmoid(prediction) > 0.5
    return (1. * (pred_mask == graph.y)).mean()


def node_mask_accuracy_graph_level(graph, prediction):
    return 1.0 * (node_mask_accuracy_node_level(graph, prediction) == 1.)


def evaluate(model, dataloader, calculators, device):
    scores = defaultdict(float)
    total_points = 0
    for data in dataloader:
        data = data.to(device)

        batched_prediction = model(data.node_fts, data.edge_attr,
                                   data.edge_index, data.reverse_idx, data.scalars, data.batch)
        for batch_idx, graph in enumerate(data.to_data_list()):
            batch_pred_idx = data.batch if model.output_from_nodes else data.batch[data.edge_index[0]]
            prediction = batched_prediction[batch_pred_idx == batch_idx]

            for calculator in calculators:
                value = calculator(graph, prediction)
                scores[calculator.__name__] += value if isinstance(value, float) else value.item()
            total_points += 1
    for calculator in calculators:
        scores[calculator.__name__] /= total_points
    return scores


class ModelSaver:
    def __init__(self, models_directory: str, model_name: str):
        Path(models_directory).mkdir(parents=True, exist_ok=True)
        model_name = "{}/{}".format(models_directory, model_name)

        self.best_vals = defaultdict(float)
        self.model_name = model_name

    def visit(self, model, metrics_stat):
        # default value is 0, greater is better
        for metric in metrics_stat:
            if metrics_stat[metric] > self.best_vals[metric]:
                self.best_vals[metric] = metrics_stat[metric]
                self.save(model, metric)
        self.save(model, "last")  # always save last visited model

    def save(self, model, suffix: str = ''):
        path = self.model_name if not suffix else self.model_name + "_" + suffix
        print("saving model: ", path)
        torch.save(model.state_dict(), path)


NODE_POINTER_METRICS = (node_pointer_score, node_pointer_all_match)
NODE_MASK_METRICS = (node_mask_accuracy_node_level, node_mask_accuracy_graph_level)

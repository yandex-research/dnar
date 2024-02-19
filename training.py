import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import metrics
import models
import pipelines
from configs import base_config
from data_utils import DataGenerator


def train(config: base_config.Config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)
    pipeline = pipelines.PIPELINES[config.pipeline_name]
    model = models.NarModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("model params count: ", total_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    model_name = "{}".format(config.pipeline_name)
    model_saver = metrics.ModelSaver(config.models_directory, model_name)

    train_data = DataGenerator(config, pipeline, 'train').as_dataloader(shuffle=True)
    val_data = DataGenerator(config, pipeline, 'val').as_dataloader()
    test_data = DataGenerator(config, pipeline, 'test').as_dataloader()

    if config.tensorboard_logs:
        writer = SummaryWriter(comment=f"-{model_name}")

    model.train()

    steps = 0
    while steps <= config.num_iterations:
        for batch in train_data:
            steps += 1

            data = batch.to(device)
            hints = (batch.mid_y, batch.mid_edge_y) if config.use_hints else None
            if hints is not None:
                pred, hint_loss = model(data.node_fts, data.edge_attr, data.edge_index,
                                        data.reverse_idx, data.scalars, data.batch, training_step=steps, hints=hints)
            else:
                pred = model(data.node_fts, data.edge_attr, data.edge_index, data.reverse_idx,
                             data.scalars, data.batch, training_step=steps, hints=hints)
                hint_loss = 0.

            output_loss = pipeline.loss(data.y, pred, data.edge_index, data.batch)
            if config.use_hints and not config.stepwise_training:
                output_loss *= 0.01  # with hints output loss is almost redundant
            loss = output_loss + hint_loss

            assert not torch.isnan(loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            if config.tensorboard_logs:
                writer.add_scalar("Loss/train", loss.detach().item(), steps)

            if steps % config.eval_each == 1:
                with torch.no_grad():
                    model.eval()
                    val_scores = metrics.evaluate(model, val_data, pipeline.metrics, device)
                    test_scores = metrics.evaluate(model, test_data, pipeline.metrics, device)
                    model.train()

                if config.tensorboard_logs:
                    for stat in val_scores:
                        writer.add_scalar(stat + "/val", val_scores[stat], steps)
                        writer.add_scalar(stat + "/test", test_scores[stat], steps)
                model_saver.visit(model, val_scores)
            if steps >= config.num_iterations:
                break
    model.eval()
    return model


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/bfs_stepwise.yaml")
    options = parser.parse_args()

    print("Train with config {}".format(options.config_path))
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_default_tensor_type(torch.DoubleTensor)

    config = base_config.read_config(options.config_path)
    model = train(config)

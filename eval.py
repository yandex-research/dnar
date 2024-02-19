import torch

import metrics
import pipelines
from configs import base_config
from data_utils import DataGenerator
from models import NarModel

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    config_path = './configs/dfs_stepwise.yaml'
    config = base_config.read_config(config_path)
    pipeline = pipelines.PIPELINES[config.pipeline_name]
    split = 'val'

    model = NarModel(config)
    model_path = config.models_directory + "/" + "dfs_last"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to(device)

    sampler = DataGenerator(config, pipeline, split).as_dataloader()

    with torch.no_grad():
        scores = metrics.evaluate(model, sampler, pipeline.metrics, device)
        print(scores)

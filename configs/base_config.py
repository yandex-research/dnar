from dataclasses import dataclass

import yaml


@dataclass
class Config:
    pipeline_name: str = None

    # --- train ---
    batch_size: int = 32
    learning_rate: float = 0.0003
    weight_decay: float = 1.0
    num_iterations: int = 1 + 50000
    eval_each: int = 500
    stepwise_training: bool = False
    use_hints: bool = True
    hint_loss_weight: bool = False
    processor_lower_t: float = 0.5
    processor_upper_t: float = 3.
    states_lower_t: float = 0.5
    states_upper_t: float = 3.
    use_noise: bool = True

    # --- data ---
    weighted_graphs: bool = False
    generate_random_numbers: bool = False
    generate_start: bool = True
    sampler_name: dict = None
    samples_count: dict = None
    problem_size: dict = None

    # --- model ---
    h: int = 128

    use_discrete_bottleneck: bool = True
    node_states_count: int = 5
    edge_states_count: int = 5
    message_top_one_choice: str = "reciever"
    position_dim: int = 0
    steps_mul: int = 1
    output_location: str = 'edge'
    update_scalars: bool = False
    temp_on_eval: float = 0.

    # --- logs, io ---
    models_directory: str = 'models'
    tensorboard_logs: bool = True


def read_config(config_path: str):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)

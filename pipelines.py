from dataclasses import dataclass
from typing import Callable, Tuple

import algorithms
import metrics
import state_algorithms


@dataclass
class Pipeline:
    algorithm: Callable
    states_algorithm: Callable
    scalars_update: Callable
    loss: Callable
    metrics: Tuple[Callable]


PIPELINES = {}

PIPELINES['bfs'] = Pipeline(
    algorithm=algorithms.bfs,
    states_algorithm=state_algorithms.bfs_states,
    scalars_update=None,
    loss=metrics.node_pointer_loss,
    metrics=metrics.NODE_POINTER_METRICS
)

PIPELINES['dfs'] = Pipeline(
    algorithm=algorithms.dfs,
    states_algorithm=state_algorithms.dfs_states,
    scalars_update=None,
    loss=metrics.node_pointer_loss,
    metrics=metrics.NODE_POINTER_METRICS
)

PIPELINES['prim'] = Pipeline(
    algorithm=algorithms.prim,
    states_algorithm=state_algorithms.prim_states,
    scalars_update=None,
    loss=metrics.node_pointer_loss,
    metrics=metrics.NODE_POINTER_METRICS
)

PIPELINES['mis'] = Pipeline(
    algorithm=algorithms.mis,
    states_algorithm=state_algorithms.mis_states,
    scalars_update=None,
    loss=metrics.node_mask_loss,
    metrics=metrics.NODE_MASK_METRICS
)

PIPELINES['dijkstra'] = Pipeline(
    algorithm=algorithms.dijkstra,
    states_algorithm=state_algorithms.dijkstra_states,
    scalars_update=state_algorithms.dijkstra_scalars_update,
    loss=metrics.node_pointer_loss,
    metrics=metrics.NODE_POINTER_METRICS
)

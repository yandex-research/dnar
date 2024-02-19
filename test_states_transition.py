import math
import unittest

import torch

from configs import base_config
from models import NarModel
from state_algorithms import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# used when node does not recieve any message, occurs only with global message choice
EMPTY_STATE = -1


def node_state_transition(model, sender_node_state, reciever_node_state, self_loop):
    if self_loop:
        assert sender_node_state == reciever_node_state
    node_states_count = model.node_states_count

    if sender_node_state != EMPTY_STATE:
        sender_ohe = torch.zeros(1, node_states_count).to(DEVICE)
        sender_ohe[0][sender_node_state] = 1.
        sender_fts = model.processor.lin_value(model.node_encoder(sender_ohe))
    else:
        sender_fts = torch.zeros(1, model.h).to(DEVICE)

    reciever_ohe = torch.zeros(1, node_states_count).to(DEVICE)
    reciever_ohe[0][reciever_node_state] = 1.
    reciever_fts = model.node_encoder(reciever_ohe)

    self_loop_fts = torch.ones_like(sender_fts) * self_loop

    new_node_fts = model.processor.node_mlp(torch.hstack([sender_fts, reciever_fts, self_loop_fts]))
    new_node_fts = model.processor.ln_nodes(new_node_fts)
    return torch.argmax(model.node_projection(new_node_fts), -1).item()


def edge_state_transition(model, sender_node_state, reciever_node_state, edge_state, alpha: int, reverse_alpha: int):
    node_states_count = model.node_states_count
    edge_states_count = model.edge_states_count

    sender_ohe = torch.zeros(1, node_states_count).to(DEVICE)
    sender_ohe[0][sender_node_state] = 1.
    sender_fts = model.processor.lin_key(model.node_encoder(sender_ohe))

    reciever_ohe = torch.zeros(1, node_states_count).to(DEVICE)
    reciever_ohe[0][reciever_node_state] = 1.
    reciever_fts = model.processor.lin_query(model.node_encoder(reciever_ohe))

    node_to_edge = model.processor.node_to_edge(torch.hstack([sender_fts, reciever_fts]))

    edge_ohe = torch.zeros(1, edge_states_count).to(DEVICE)
    edge_ohe[0][edge_state] = 1.
    edge_fts = model.edge_encoder(edge_ohe)

    alpha_fts = torch.ones_like(edge_fts) * alpha
    reverse_alpha_fts = torch.ones_like(edge_fts) * reverse_alpha

    new_edge_fts = model.processor.edge_mlp(torch.hstack([edge_fts, node_to_edge, reverse_alpha_fts, alpha_fts]))
    new_edge_fts = model.processor.ln_edges(new_edge_fts)
    return torch.argmax(model.edge_projection(new_edge_fts), -1).item()


def states_attention_range(model, sender_node_state, reciever_node_state, edge_state):
    node_states_count = model.node_states_count
    edge_states_count = model.edge_states_count

    sender_ohe = torch.zeros(1, node_states_count).to(DEVICE)
    sender_ohe[0][sender_node_state] = 1.
    sender_fts = model.processor.lin_key(model.node_encoder(sender_ohe))

    reciever_ohe = torch.zeros(1, node_states_count).to(DEVICE)
    reciever_ohe[0][reciever_node_state] = 1.
    reciever_fts = model.processor.lin_query(model.node_encoder(reciever_ohe))

    edge_ohe = torch.zeros(1, edge_states_count).to(DEVICE)
    edge_ohe[0][edge_state] = 1.
    edge_fts = model.processor.edge_key(model.edge_encoder(edge_ohe))

    min_scalar = 0. * model.processor.scalar_w
    max_scalar = 1. * model.processor.scalar_w
    if min_scalar > max_scalar:
        min_scalar, max_scalar = max_scalar, min_scalar

    min_alpha = (reciever_fts * (sender_fts + edge_fts) + min_scalar).sum(-1).item() / math.sqrt(model.h)
    max_alpha = (reciever_fts * (sender_fts + edge_fts) + max_scalar).sum(-1).item() / math.sqrt(model.h)
    return min_alpha, max_alpha


class TestBFSModel(unittest.TestCase):
    def setUp(self):
        config_path = './configs/bfs_stepwise.yaml'
        config = base_config.read_config(config_path)

        model = NarModel(config=config)
        model_path = config.models_directory + "/" + "bfs_last"
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = model.to(DEVICE)

    def test_node_states_transition(self):
        self.assertEqual(node_state_transition(self.model, BFS_DISCOVERED, BFS_NOT_DISCOVERED, 0), BFS_DISCOVERED)
        self.assertEqual(node_state_transition(
            self.model, BFS_NOT_DISCOVERED, BFS_NOT_DISCOVERED, 0), BFS_NOT_DISCOVERED)
        self.assertEqual(node_state_transition(
            self.model, BFS_NOT_DISCOVERED, BFS_NOT_DISCOVERED, 1), BFS_NOT_DISCOVERED)

        self.assertEqual(node_state_transition(self.model, BFS_DISCOVERED, BFS_DISCOVERED, 0), BFS_DISCOVERED)
        self.assertEqual(node_state_transition(self.model, BFS_DISCOVERED, BFS_DISCOVERED, 1), BFS_DISCOVERED)
        self.assertEqual(node_state_transition(self.model, BFS_NOT_DISCOVERED, BFS_DISCOVERED, 0), BFS_DISCOVERED)

    def test_edge_states_transition(self):
        self.assertEqual(edge_state_transition(
            self.model, BFS_NOT_DISCOVERED, BFS_DISCOVERED, BFS_EDGE_NONE, 0, 1), BFS_EDGE_POINTER)
        self.assertEqual(edge_state_transition(
            self.model, BFS_NOT_DISCOVERED, BFS_DISCOVERED, BFS_EDGE_NONE, 0, 0), BFS_EDGE_NONE)

        self.assertEqual(edge_state_transition(self.model, BFS_NOT_DISCOVERED,
                         BFS_NOT_DISCOVERED, BFS_EDGE_NONE, 0, 0), BFS_EDGE_NONE)
        self.assertEqual(edge_state_transition(self.model, BFS_NOT_DISCOVERED,
                         BFS_NOT_DISCOVERED, BFS_EDGE_NONE, 1, 0), BFS_EDGE_NONE)
        self.assertEqual(edge_state_transition(self.model, BFS_NOT_DISCOVERED,
                         BFS_NOT_DISCOVERED, BFS_EDGE_NONE, 0, 1), BFS_EDGE_NONE)
        self.assertEqual(edge_state_transition(self.model, BFS_NOT_DISCOVERED,
                         BFS_NOT_DISCOVERED, BFS_EDGE_NONE, 1, 1), BFS_EDGE_NONE)
        self.assertEqual(edge_state_transition(
            self.model, BFS_DISCOVERED, BFS_DISCOVERED, BFS_EDGE_NONE, 0, 0), BFS_EDGE_NONE)
        self.assertEqual(edge_state_transition(
            self.model, BFS_DISCOVERED, BFS_DISCOVERED, BFS_EDGE_NONE, 1, 0), BFS_EDGE_NONE)
        self.assertEqual(edge_state_transition(
            self.model, BFS_DISCOVERED, BFS_DISCOVERED, BFS_EDGE_NONE, 0, 1), BFS_EDGE_NONE)
        self.assertEqual(edge_state_transition(
            self.model, BFS_DISCOVERED, BFS_DISCOVERED, BFS_EDGE_NONE, 1, 1), BFS_EDGE_NONE)

        self.assertEqual(edge_state_transition(self.model, BFS_DISCOVERED,
                         BFS_DISCOVERED, BFS_EDGE_POINTER, 0, 0), BFS_EDGE_POINTER)
        self.assertEqual(edge_state_transition(self.model, BFS_DISCOVERED,
                         BFS_DISCOVERED, BFS_EDGE_POINTER, 1, 0), BFS_EDGE_POINTER)
        self.assertEqual(edge_state_transition(self.model, BFS_DISCOVERED,
                         BFS_DISCOVERED, BFS_EDGE_POINTER, 0, 1), BFS_EDGE_POINTER)
        self.assertEqual(edge_state_transition(self.model, BFS_DISCOVERED,
                         BFS_DISCOVERED, BFS_EDGE_POINTER, 1, 1), BFS_EDGE_POINTER)

    def test_attention_discovered_vs_not_for_not(self):
        MI = states_attention_range(self.model, BFS_DISCOVERED, BFS_NOT_DISCOVERED, BFS_EDGE_NONE)
        LI = states_attention_range(self.model, BFS_NOT_DISCOVERED, BFS_NOT_DISCOVERED, BFS_EDGE_NONE)
        self.assertTrue(LI[1] < MI[0], (LI, MI))

    def test_non_pointer_becomes_pointer_only_on_reverse_discovery(self):
        for sender_state in (BFS_DISCOVERED, BFS_NOT_DISCOVERED):
            for reciever_state in (BFS_DISCOVERED, BFS_NOT_DISCOVERED):
                for alphas in ((0, 0), (0, 1), (1, 0), (1, 1)):
                    non_pointer_becomes_pointer = edge_state_transition(
                        self.model, sender_state, reciever_state, BFS_EDGE_NONE, *alphas) == BFS_EDGE_POINTER
                    if non_pointer_becomes_pointer:
                        self.assertTrue(
                            sender_state == BFS_NOT_DISCOVERED and reciever_state == BFS_DISCOVERED and alphas[1] == 1)


class TestDFSModel(unittest.TestCase):
    def setUp(self):
        config_path = './configs/dfs_stepwise.yaml'
        config = base_config.read_config(config_path)

        model = NarModel(config=config)
        model_path = config.models_directory + "/" + "dfs_last"
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = model.to(DEVICE)

    def test_node_states_transition(self):
        self.assertEqual(node_state_transition(
            self.model, DFS_ON_TOP_OF_THE_STACK, DFS_NOT_IN_THE_STACK, 0), DFS_ON_TOP_OF_THE_STACK)
        self.assertEqual(node_state_transition(
            self.model, DFS_ON_TOP_OF_THE_STACK, DFS_ON_TOP_OF_THE_STACK, 1), DFS_PREEND)
        self.assertEqual(node_state_transition(self.model, DFS_PREEND, DFS_ON_THE_STACK, 0), DFS_ON_TOP_OF_THE_STACK)
        self.assertEqual(node_state_transition(self.model, DFS_PREEND, DFS_PREEND, 1), DFS_END)
        self.assertEqual(node_state_transition(self.model, EMPTY_STATE, DFS_PREEND, 0), DFS_END)

    def test_edge_states_transition(self):
        self.assertEqual(edge_state_transition(self.model, DFS_ON_TOP_OF_THE_STACK,
                         DFS_NOT_IN_THE_STACK, DFS_EDGE_NONE, 1, 0), DFS_EDGE_NONE)
        self.assertEqual(edge_state_transition(self.model, DFS_NOT_IN_THE_STACK,
                         DFS_ON_TOP_OF_THE_STACK, DFS_EDGE_NONE, 0, 1), DFS_EDGE_POINTER)
        self.assertEqual(edge_state_transition(self.model, DFS_PREEND,
                         DFS_ON_THE_STACK, DFS_EDGE_POINTER, 1, 0), DFS_EDGE_POINTER)

    def test_attention_new_vs_stack_for_top(self):
        MI = states_attention_range(self.model, DFS_ON_TOP_OF_THE_STACK, DFS_NOT_IN_THE_STACK, DFS_EDGE_NONE)
        LI = states_attention_range(self.model, DFS_ON_TOP_OF_THE_STACK, DFS_ON_THE_STACK, DFS_EDGE_NONE)
        self.assertTrue(LI[1] < MI[0], (LI, MI))

    def test_attention_new_vs_end_for_top(self):
        MI = states_attention_range(self.model, DFS_ON_TOP_OF_THE_STACK, DFS_NOT_IN_THE_STACK, DFS_EDGE_NONE)
        LI = states_attention_range(self.model, DFS_ON_TOP_OF_THE_STACK, DFS_END, DFS_EDGE_NONE)
        self.assertTrue(LI[1] < MI[0], (LI, MI))

    def test_attentino_top_vs_stack_for_top(self):
        MI = states_attention_range(self.model, DFS_ON_TOP_OF_THE_STACK, DFS_ON_TOP_OF_THE_STACK, DFS_EDGE_NONE)
        LI = states_attention_range(self.model, DFS_ON_TOP_OF_THE_STACK, DFS_ON_THE_STACK, DFS_EDGE_NONE)
        self.assertTrue(LI[1] < MI[0], (LI, MI))

    def test_attention_parent_node_vs_end_for_preend(self):
        MI = states_attention_range(self.model, DFS_PREEND, DFS_ON_THE_STACK, DFS_EDGE_POINTER)
        LI = states_attention_range(self.model, DFS_PREEND, DFS_END, DFS_EDGE_NONE)
        self.assertTrue(LI[1] < MI[0], (LI, MI))

    def test_attention_parent_node_vs_non_parent_stack_for_preend(self):
        MI = states_attention_range(self.model, DFS_PREEND, DFS_ON_THE_STACK, DFS_EDGE_POINTER)
        LI = states_attention_range(self.model, DFS_PREEND, DFS_ON_THE_STACK, DFS_EDGE_NONE)
        self.assertTrue(LI[1] < MI[0], (LI, MI))


class TestDijkstraModel(unittest.TestCase):
    def setUp(self):
        config_path = './configs/dijkstra_stepwise.yaml'
        config = base_config.read_config(config_path)

        model = NarModel(config=config)
        model_path = config.models_directory + "/" + "dijkstra_last"
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = model.to(DEVICE)

    def test_node_states_transition(self):
        self.assertEqual(node_state_transition(
            self.model, DIJKSTRA_IN_THE_TREE, DIJKSTRA_NOT_IN_THE_TREE, 0), DIJKSTRA_IN_THE_TREE)
        self.assertEqual(node_state_transition(
            self.model, DIJKSTRA_IN_THE_TREE, DIJKSTRA_IN_THE_TREE, 0), DIJKSTRA_IN_THE_TREE)
        self.assertEqual(node_state_transition(
            self.model, DIJKSTRA_IN_THE_TREE, DIJKSTRA_IN_THE_TREE, 1), DIJKSTRA_IN_THE_TREE)
        self.assertEqual(node_state_transition(
            self.model, EMPTY_STATE, DIJKSTRA_NOT_IN_THE_TREE, 0), DIJKSTRA_NOT_IN_THE_TREE)
        self.assertEqual(node_state_transition(
            self.model, EMPTY_STATE, DIJKSTRA_IN_THE_TREE, 0), DIJKSTRA_IN_THE_TREE)

    def test_edge_states_transition(self):
        self.assertEqual(edge_state_transition(self.model, DIJKSTRA_IN_THE_TREE,
                         DIJKSTRA_NOT_IN_THE_TREE, DIJKSTRA_EDGE_NONE, 1, 0), DIJKSTRA_EDGE_NONE)
        self.assertEqual(edge_state_transition(self.model, DIJKSTRA_IN_THE_TREE,
                         DIJKSTRA_NOT_IN_THE_TREE, DIJKSTRA_EDGE_NONE, 0, 0), DIJKSTRA_EDGE_NONE)
        self.assertEqual(edge_state_transition(self.model, DIJKSTRA_NOT_IN_THE_TREE,
                         DIJKSTRA_IN_THE_TREE, DIJKSTRA_EDGE_NONE, 0, 1), DIJKSTRA_EDGE_PREPOINTER)
        self.assertEqual(edge_state_transition(self.model, DIJKSTRA_IN_THE_TREE,
                         DIJKSTRA_IN_THE_TREE, DIJKSTRA_EDGE_PREPOINTER, 0, 0), DIJKSTRA_EDGE_POINTER)
        self.assertEqual(edge_state_transition(self.model, DIJKSTRA_IN_THE_TREE,
                         DIJKSTRA_IN_THE_TREE, DIJKSTRA_EDGE_PREPOINTER, 0, 1), DIJKSTRA_EDGE_POINTER)
        self.assertEqual(edge_state_transition(self.model, DIJKSTRA_IN_THE_TREE,
                         DIJKSTRA_IN_THE_TREE, DIJKSTRA_EDGE_POINTER, 0, 0), DIJKSTRA_EDGE_POINTER)
        self.assertEqual(edge_state_transition(self.model, DIJKSTRA_IN_THE_TREE,
                         DIJKSTRA_IN_THE_TREE, DIJKSTRA_EDGE_POINTER, 1, 0), DIJKSTRA_EDGE_POINTER)
        self.assertEqual(edge_state_transition(self.model, DIJKSTRA_IN_THE_TREE,
                         DIJKSTRA_IN_THE_TREE, DIJKSTRA_EDGE_POINTER, 0, 1), DIJKSTRA_EDGE_POINTER)

    def test_attention_not_tree_vs_tree_for_tree(self):
        MI = states_attention_range(
            self.model, DIJKSTRA_IN_THE_TREE, DIJKSTRA_NOT_IN_THE_TREE, DIJKSTRA_EDGE_NONE)
        for edge_state in (DIJKSTRA_EDGE_NONE, DIJKSTRA_EDGE_POINTER, DIJKSTRA_EDGE_PREPOINTER):
            LI = states_attention_range(
                self.model, DIJKSTRA_IN_THE_TREE, DIJKSTRA_IN_THE_TREE, edge_state)
            self.assertTrue(LI[1] < MI[0], (LI, MI))

    def test_attention_tree_vs_no_tree_for_not_tree(self):
        MI = states_attention_range(
            self.model, DIJKSTRA_IN_THE_TREE, DIJKSTRA_NOT_IN_THE_TREE, DIJKSTRA_EDGE_NONE)
        LI = states_attention_range(
            self.model, DIJKSTRA_NOT_IN_THE_TREE, DIJKSTRA_NOT_IN_THE_TREE, DIJKSTRA_EDGE_NONE)
        self.assertTrue(LI[1] < MI[0], (LI, MI))


class TestPrimModel(unittest.TestCase):
    def setUp(self):
        config_path = './configs/prim_stepwise.yaml'
        config = base_config.read_config(config_path)

        model = NarModel(config=config)
        model_path = config.models_directory + "/" + "prim_last"
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = model.to(DEVICE)

    def test_node_states_transition(self):
        self.assertEqual(node_state_transition(self.model, PRIM_IN_THE_TREE, PRIM_NOT_IN_THE_TREE, 0), PRIM_IN_THE_TREE)
        self.assertEqual(node_state_transition(self.model, PRIM_IN_THE_TREE, PRIM_IN_THE_TREE, 0), PRIM_IN_THE_TREE)
        self.assertEqual(node_state_transition(self.model, PRIM_IN_THE_TREE, PRIM_IN_THE_TREE, 1), PRIM_IN_THE_TREE)
        self.assertEqual(node_state_transition(self.model, EMPTY_STATE, PRIM_NOT_IN_THE_TREE, 0), PRIM_NOT_IN_THE_TREE)
        self.assertEqual(node_state_transition(self.model, EMPTY_STATE, PRIM_IN_THE_TREE, 0), PRIM_IN_THE_TREE)

    def test_edge_states_transition(self):
        self.assertEqual(edge_state_transition(self.model, PRIM_IN_THE_TREE,
                         PRIM_NOT_IN_THE_TREE, PRIM_EDGE_NONE, 1, 0), PRIM_EDGE_NONE)
        self.assertEqual(edge_state_transition(self.model, PRIM_NOT_IN_THE_TREE,
                         PRIM_IN_THE_TREE, PRIM_EDGE_NONE, 0, 1), PRIM_EDGE_POINTER)

        self.assertEqual(edge_state_transition(self.model, PRIM_NOT_IN_THE_TREE,
                         PRIM_IN_THE_TREE, PRIM_EDGE_NONE, 0, 0), PRIM_EDGE_NONE)
        self.assertEqual(edge_state_transition(
            self.model, PRIM_IN_THE_TREE, PRIM_IN_THE_TREE, PRIM_EDGE_NONE, 0, 0), PRIM_EDGE_NONE)
        self.assertEqual(edge_state_transition(
            self.model, PRIM_IN_THE_TREE, PRIM_IN_THE_TREE, PRIM_EDGE_NONE, 0, 1), PRIM_EDGE_NONE)
        self.assertEqual(edge_state_transition(
            self.model, PRIM_IN_THE_TREE, PRIM_IN_THE_TREE, PRIM_EDGE_NONE, 1, 0), PRIM_EDGE_NONE)

    def test_attention_not_tree_vs_tree_for_tree(self):
        MI = states_attention_range(self.model, PRIM_IN_THE_TREE, PRIM_NOT_IN_THE_TREE, PRIM_EDGE_NONE)
        for edge_state in (PRIM_EDGE_NONE, PRIM_EDGE_POINTER):
            LI = states_attention_range(self.model, PRIM_IN_THE_TREE, PRIM_IN_THE_TREE, edge_state)
            self.assertTrue(LI[1] < MI[0], (LI, MI))

    def test_attention_tree_vs_no_tree_for_not_tree(self):
        MI = states_attention_range(self.model, PRIM_IN_THE_TREE, PRIM_NOT_IN_THE_TREE, PRIM_EDGE_NONE)
        LI = states_attention_range(self.model, PRIM_NOT_IN_THE_TREE, PRIM_NOT_IN_THE_TREE, PRIM_EDGE_NONE)
        self.assertTrue(LI[1] < MI[0], (LI, MI))


class TestMISModel(unittest.TestCase):
    def setUp(self):
        config_path = './configs/mis_stepwise.yaml'
        config = base_config.read_config(config_path)

        model = NarModel(config=config)
        model_path = config.models_directory + "/" + "mis_last"
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = model.to(DEVICE)

    def test_states_transition(self):
        self.assertEqual(node_state_transition(self.model, MIS_ALIVE, MIS_ALIVE, 1), MIS_IN_MIS)
        self.assertEqual(node_state_transition(self.model, MIS_ALIVE, MIS_ALIVE, 0), MIS_WAIT)

        self.assertEqual(node_state_transition(self.model, MIS_WAIT, MIS_IN_MIS, 0), MIS_IN_MIS)
        self.assertEqual(node_state_transition(self.model, MIS_IN_MIS, MIS_IN_MIS, 0), MIS_IN_MIS)
        self.assertEqual(node_state_transition(self.model, MIS_IN_MIS, MIS_IN_MIS, 1), MIS_IN_MIS)

        self.assertEqual(node_state_transition(self.model, MIS_WAIT, MIS_WAIT, 0), MIS_ALIVE)
        self.assertEqual(node_state_transition(self.model, MIS_WAIT, MIS_WAIT, 1), MIS_ALIVE)
        self.assertEqual(node_state_transition(self.model, MIS_DEAD_NOT_IN_MIS, MIS_WAIT, 0), MIS_ALIVE)
        self.assertEqual(node_state_transition(self.model, MIS_IN_MIS, MIS_WAIT, 0), MIS_DEAD_NOT_IN_MIS)

        self.assertEqual(node_state_transition(
            self.model, MIS_DEAD_NOT_IN_MIS, MIS_DEAD_NOT_IN_MIS, 0), MIS_DEAD_NOT_IN_MIS)
        self.assertEqual(node_state_transition(
            self.model, MIS_DEAD_NOT_IN_MIS, MIS_DEAD_NOT_IN_MIS, 1), MIS_DEAD_NOT_IN_MIS)

    def test_mis_vs_wait_for_wait(self):
        MI = states_attention_range(self.model, MIS_IN_MIS, MIS_WAIT, MIS_EDGE_NONE)
        LI = states_attention_range(self.model, MIS_WAIT, MIS_WAIT, MIS_EDGE_NONE)
        self.assertTrue(LI[1] < MI[0], (LI, MI))

    def test_mis_vs_dead_for_wait(self):
        MI = states_attention_range(self.model, MIS_IN_MIS, MIS_WAIT, MIS_EDGE_NONE)
        LI = states_attention_range(self.model, MIS_DEAD_NOT_IN_MIS, MIS_WAIT, MIS_EDGE_NONE)
        self.assertTrue(LI[1] < MI[0], (LI, MI))

    def test_alive_vs_dead_for_alive(self):
        MI = states_attention_range(self.model, MIS_ALIVE, MIS_ALIVE, MIS_EDGE_NONE)
        LI = states_attention_range(self.model, MIS_DEAD_NOT_IN_MIS, MIS_ALIVE, MIS_EDGE_NONE)
        self.assertTrue(LI[1] < MI[0], (LI, MI))


if __name__ == "__main__":
    print("device: ", DEVICE)

    with torch.no_grad():
        unittest.main()

import logging
import os
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Set, Generator

import numpy as np
import torch
import networkx as nx
import wandb

from replay import ReplayBufferSafeRL

from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


class CausalDynamicModel(ABC):
    """
    A class to represent a safe related causal DAG.
    """

    def __init__(self, state_dict: Dict, action_dim: int):
        self.state_dict = state_dict
        self.action_dim = action_dim

        nodes, nodes_dict = self.init_nodes()
        self._pdag = Dag(nodes=nodes)
        self.bg_knowledge = self.init_background_knowledge()
        self.nodes_dict = nodes_dict

    def init_nodes(self) -> List[GraphNode]:
        nodes = []
        state_nodes = self.state_dict["reward"] + self.state_dict["cost"]

        curr_state_nodes = [GraphNode(f"y_{i}") for i in range(state_nodes)]
        next_state_nodes = [GraphNode(f"nexty_{i}") for i in range(state_nodes)]
        action_node = [GraphNode(f"action_{i}") for i in range(self.action_dim)]
        reward_node = [GraphNode("reward")]
        cost_node = [GraphNode("cost")]
        nodes = curr_state_nodes + action_node + next_state_nodes + reward_node + cost_node

        nodes_dict = dict(y=curr_state_nodes, next_y=next_state_nodes, action=action_node, reward=reward_node, cost=cost_node)

        return nodes, nodes_dict

    def init_background_knowledge(self) -> BackgroundKnowledge:
        bg_knowledge = BackgroundKnowledge()
        # y_{t} -> y_{t} or y_{t+1} -> y_{t+1}, or a_{t+1} -> a_{t+1}
        bg_knowledge.add_forbidden_by_pattern("y", "y")
        bg_knowledge.add_forbidden_by_pattern("nexty", "nexty")
        bg_knowledge.add_forbidden_by_pattern("action", "action")
        # y_{t+1} -> y_{t}
        bg_knowledge.add_forbidden_by_pattern("nexty", "y")
        # r_{t+1} -> y_{t+1} or r_{t+1} -> y_{t}
        bg_knowledge.add_forbidden_by_pattern("reward", "nexty")
        bg_knowledge.add_forbidden_by_pattern("reward", "y")
        # c_{t+1} -> y_{t+1} or c_{t+1} -> y_{t}
        bg_knowledge.add_forbidden_by_pattern("cost", "nexty")
        bg_knowledge.add_forbidden_by_pattern("cost", "y")
        return bg_knowledge

    def compute(self, buffer: ReplayBufferSafeRL, wm, seed) -> Dict:
        metrics = {}
        num_samples = 512
        target_device = wm.device
        wm.eval()

        eval_rng = np.random.Generator(np.random.PCG64(seed))

        idx = buffer.sample_idx(num_samples, eval_rng)
        keys = ("o", "a", "next_o", "next_r", "next_c")
        data = buffer.get(idx, *keys)

        (o, a, next_o, next_r, next_c) = [item.to(target_device) for item in data]

        with torch.no_grad():
            o = wm.preprocess(o)
            next_o = wm.preprocess(next_o)
            y = wm.encoder(o)
            next_y = wm.encoder(next_o)

        fci_input, node_names = self.preprocess_data(y, a, next_y, next_r, next_c, 32)
        new_edges = self._fci_compute(fci_input, node_names)
        metrics["new_edges"] = new_edges

        return metrics

    def evaluate(self) -> Dict:
        metrics = {}
        output_graph = self.output_graph()
        if output_graph is not None:
            metrics["graph"] = wandb.Image(output_graph, mode="RGB")
        else:
            metrics["graph"] = ""

        return metrics

    def preprocess_data(
        self,
        y: torch.Tensor,
        actions: torch.Tensor,
        next_y: torch.Tensor,
        rewards: torch.Tensor,
        costs: torch.Tensor,
        num_variables: int,
    ) -> Tuple[torch.Tensor, List]:
        y_r, _, y_c, _ = torch.split(y, [self.state_dict["reward"], self.state_dict["reward"], self.state_dict["cost"], self.state_dict["cost"]], dim=-1)

        next_y_r, _, next_y_c, _ = torch.split(next_y, [self.state_dict["reward"], self.state_dict["reward"], self.state_dict["cost"], self.state_dict["cost"]], dim=-1)

        y = torch.cat((y_r, y_c), dim=-1)
        next_y = torch.cat((next_y_r, next_y_c), dim=-1)
        idx = torch.randperm(y.shape[1])[:num_variables]
        y = y[:, idx]
        next_y = next_y[:, idx]
        actions = actions[:, -1, :]
        rewards = rewards.unsqueeze(-1)
        costs = costs.unsqueeze(-1)
        data = torch.cat((y, actions, next_y, rewards, costs), axis=-1)  # (n_samples, n_feature)

        node_names = (
            [self.nodes_dict["y"][i].name for i in idx]
            + [node.name for node in self.nodes_dict["action"]]
            + [self.nodes_dict["next_y"][i].name for i in idx]
            + [node.name for node in self.nodes_dict["reward"]]
            + [node.name for node in self.nodes_dict["cost"]]
        )
        return data.reshape(-1, data.shape[-1]).numpy().astype(np.float32), node_names

    def _fci_compute(self, data: np.ndarray, node_names: Dict) -> int:
        """
        Testing only: Compute the FCI algorithm
        """

        data = data.astype(np.float32)
        graph, edges = fci(data, depth=2, max_path_length=3, verbose=False, background_knowledge=self.bg_knowledge, node_names=node_names)

        ct = 0
        for item in edges:
            node1, node2 = item.get_node1(), item.get_node2()
            # find index
            pdag_node1, pdag_node2 = self._pdag.get_node(node1.name), self._pdag.get_node(node2.name)

            sanity_case = self.sanity_check(pdag_node1, pdag_node2)

            if sanity_case == 0:
                new_edge = Edge(pdag_node1, pdag_node2, end1=item.get_endpoint1(), end2=item.get_endpoint2())
                # if "dd" in new_edge.properties:
                #     continue
                flag = self._pdag.add_edge(new_edge)
                ct += 1
                if not flag:
                    print("Error: Edge already exists")

        logging.info(f"Graph added: {ct} edges.")
        return ct

    def sanity_check(self, node1, node2) -> int:
        node1_type = node1.name.split("_")[0]
        node2_type = node2.name.split("_")[0]
        # y_{t} -> y_{t} or y_{t+1} -> y_{t+1}, or a_{t+1} -> a_{t+1}
        if (node1_type == "y" and node2_type == "y") or (node1_type == "nexty" and node2_type == "nexty") or (node1_type == "action" and node2_type == "action"):
            return 1
        # y_{t+1} -> y_{t}
        elif node1_type == "nexty" and node2_type == "y":
            return 2
        # r_{t+1} -> y_{t+1} or r_{t+1} -> y_{t}
        elif node1_type == "reward" and (node2_type == "nexty" or node2_type == "y"):
            return 3
        # c_{t+1} -> y_{t+1} or c_{t+1} -> y_{t}
        elif node1_type == "cost" and (node2_type == "nexty" or node2_type == "y"):
            return 3
        else:
            return 0

    def output_graph(self) -> str:
        if self._pdag is not None:
            current_logging_dir = wandb.run.dir
            logging_path = os.path.join(current_logging_dir, "causal_graph")
            pdy = GraphUtils.to_pydot(self._pdag)
            pdy.write_png(logging_path)
        else:
            logging_path = None
        return logging_path

    def _get_v_structure(self, node):
        return None

    @property
    def nodes(self) -> List:
        if self._pdag is None:
            return []
        return self._pdag.get_nodes()

    @property
    def node_names(self) -> List[str]:
        names = [n.name for n in self.nodes]
        return names

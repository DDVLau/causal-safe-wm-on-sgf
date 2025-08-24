import logging
import os
from abc import ABC
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import wandb

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.search.ConstraintBased.FCI import fci


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

    def compute(self, buffer, wm, seed) -> Dict:
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
        num_edges = self._fci_compute(fci_input, node_names)
        metrics["num_edges"] = num_edges

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
        return data.reshape(-1, data.shape[-1]).cpu().numpy().astype(np.float32), node_names

    def _fci_compute(self, data: np.ndarray, node_names: Dict) -> int:
        """
        Testing only: Compute the FCI algorithm
        """

        data = data.astype(np.float32)
        graph, edges = fci(data, depth=2, max_path_length=3, verbose=False, background_knowledge=self.bg_knowledge, node_names=node_names)

        ct = 0
        ct_nl, ct_dd, ct_pl, ct_pd = 0, 0, 0, 0
        for e in edges:
            node1, node2 = e.get_node1(), e.get_node2()
            pdag_node1, pdag_node2 = self._pdag.get_node(node1.name), self._pdag.get_node(node2.name)
            new_edge = Edge(pdag_node1, pdag_node2, end1=e.get_endpoint1(), end2=e.get_endpoint2())
            sanity_case = self.sanity_check(pdag_node1, pdag_node2)

            if sanity_case == 0:
                if len(e.properties) > 0:
                    for p in e.properties:
                        if "dd" == p.name:
                            ct_dd += 1
                        elif "nl" == p.name:
                            ct_nl += 1
                        elif "pl" == p.name:
                            ct_pl += 1
                        elif "pd" == p.name:
                            ct_pd += 1
                flag = self._pdag.add_edge(new_edge)
            elif sanity_case == 1:
                # TODO: common factors. This should be checked again
                if e.get_numerical_endpoint1() == e.get_numerical_endpoint2() == Endpoint.ARROW:
                    flag = self._pdag.add_edge(new_edge)
            if not flag:
                ct += 1

        logging.info(
            f"Total edges: {len(edges)}.\nEdge not added: {ct}.\n  No latent confounders: {ct_nl}\n  Definitely direct: {ct_dd}\n  Possibly latent confounders: {ct_pl}\n  Possibly direct:{ct_pd}"
        )
        return self._pdag.get_num_edges()

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

    def output_graph(self, num_y_nodes=100) -> Optional[str]:
        def create_subgraph(nodes, graph):
            subgraph = Dag(nodes)
            _graph = graph.graph

            for i in reversed(range(graph.num_vars)):
                if not (graph.nodes[i] in nodes):
                    _graph = np.delete(_graph, (i), axis=0)
            for i in reversed(range(graph.num_vars)):
                if not (graph.nodes[i] in nodes):
                    _graph = np.delete(_graph, (i), axis=1)
            subgraph.graph = _graph
            subgraph.reconstitute_dpath(subgraph.get_graph_edges())
            return subgraph

        if self._pdag is not None:
            logging_path = os.path.join(wandb.run.dir, "causal_graph.jpg")

            next_y_nodes = [node for node in self.nodes if node.name.startswith("nexty")]
            next_y_node_degrees = [(node, self._pdag.get_degree(node)) for node in next_y_nodes]
            top_next_y_nodes = sorted(next_y_node_degrees, key=lambda x: x[1], reverse=True)[:num_y_nodes]
            top_next_y_nodes = [n[0] for n in top_next_y_nodes]

            y_nodes = [self._pdag.get_node(f"y_{node.name.split('_')[1]}") for node in top_next_y_nodes]
            final_node_set = y_nodes + self.nodes_dict["action"] + top_next_y_nodes + self.nodes_dict["reward"] + self.nodes_dict["cost"]

            filtered_dag = create_subgraph(final_node_set, self._pdag)
            pdy = GraphUtils.to_pydot(filtered_dag)
            # TODO: change the style of plotting
            pdy.set_graph_defaults(
                layout="dot",  # or 'neato', 'fdp', 'circo' for different layouts
                rankdir="LR",  # Left-to-right layout
                ranksep="2.0",  # Increase spacing between ranks
                nodesep="1.5",  # Increase spacing between nodes
                concentrate="true",  # Merge multi-edges
                splines="ortho",  # Orthogonal edge routing
            )
            pdy.write_png(logging_path)
        else:
            logging_path = None
        return logging_path

    def _get_v_structure(self, node):
        return None

    def get_parents_siblings_set(self, node) -> Tuple[List]:
        if node.name.startswith("y"):
            parents, siblings = [], []
        else:
            # nexty_{*}, cost, reward
            parents = self._pdag.get_parents(node)
            siblings = [self._pdag.get_children(item) for item in parents]
        return parents, siblings

    def background(self) -> Dict:
        """"""
        bg_dict = {}
        for node in self.nodes_dict["next_y"]:
            parents, siblings = self.get_parents_siblings_set(node)
            parents_idx = [int(p.name.split("_")[1]) for p in parents]
            siblings_idx = [int(s.name.split("_")[1]) for s in siblings]
            bg_dict[node.name] = dict(parents_idx=parents_idx, siblings_idx=siblings_idx)
        return bg_dict

    @property
    def nodes(self) -> List:
        if self._pdag is None:
            return []
        return self._pdag.get_nodes()

    @property
    def node_names(self) -> List[str]:
        names = [n.name for n in self.nodes]
        return names

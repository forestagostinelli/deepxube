from typing import List, Any, Optional, Type, Tuple
from deepxube.base.domain import Action, NodesSupervisable, EdgesSupervisable, EdgesSampleable
from deepxube.base.pathfinding import Instance, InstanceNodeStatic, InstanceEdgeStatic, Node, EdgeQ, PathFindNodeStatic, PathFindEdgeStatic, PathFindSup
from deepxube.factories.pathfinding_factory import pathfinding_factory
import time


class InstanceSup(Instance):
    def frontier_size(self) -> int:
        raise NotImplementedError

    def record_goal(self, nodes: List[Node]) -> None:
        raise NotImplementedError

    def finished(self) -> bool:
        return self.itr > 0


class InstanceNodeSup(InstanceNodeStatic, InstanceSup):
    def __init__(self, *args: Any, path_cost_sup: Optional[float] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        assert path_cost_sup is not None
        self.path_cost_sup: float = path_cost_sup

    def filter_expanded_nodes(self, nodes: List[Node]) -> List[Node]:
        raise NotImplementedError

    def push_pop_nodes(self, nodes: List[Node], costs: List[float]) -> List[Node]:
        raise NotImplementedError


class InstanceEdgeSup(InstanceEdgeStatic, InstanceSup):
    def __init__(self, *args: Any, action: Optional[Action] = None, path_cost_sup: Optional[float] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        assert action is not None
        self.action: Action = action
        assert path_cost_sup is not None
        self.path_cost_sup: float = path_cost_sup

    def filter_popped_nodes(self, nodes: List[Node]) -> List[Node]:
        raise NotImplementedError

    def push_pop_edges(self, edges: List[EdgeQ], costs: List[float]) -> List[EdgeQ]:
        raise NotImplementedError


@pathfinding_factory.register_class("sup_v")
class PathFindNodeSup(PathFindNodeStatic[NodesSupervisable, Any, InstanceNodeSup], PathFindSup[NodesSupervisable, InstanceNodeSup]):
    @staticmethod
    def domain_type() -> Type[NodesSupervisable]:
        return NodesSupervisable

    @staticmethod
    def description() -> str:
        return "Labels nodes"

    def step(self, verbose: bool = False) -> Tuple[List[Node], List[EdgeQ]]:
        nodes: List[Node] = []
        for instance in self.instances:
            node_root: Node = instance.root_node
            node_root.heuristic = instance.path_cost_sup
            node_root.backup_val = instance.path_cost_sup
            nodes.append(node_root)
            instance.add_nodes_popped([node_root])
            instance.itr += 1
        start_time = time.time()
        # self.set_is_solved(nodes)
        self.times.record_time("is_solved", time.time() - start_time)

        return nodes, []

    def _compute_costs(self, instances: List[InstanceNodeSup], nodes_by_inst: List[List[Node]]) -> List[List[float]]:
        raise NotImplementedError

    def make_instances_sup(self, steps_gen: List[int], inst_infos: Optional[List[Any]]) -> List[InstanceNodeSup]:
        # get nodes and labels
        start_time = time.time()
        states_start, goals, labels = self.domain.samp_nodes_and_labels(steps_gen)
        self.times.record_time("get_sup", time.time() - start_time)

        # make instances
        nodes_root: List[Node] = self._create_root_nodes(states_start, goals)

        start_time = time.time()
        if inst_infos is None:
            inst_infos = [None for _ in states_start]

        instances: List[InstanceNodeSup] = []
        for node_root, inst_info, label in zip(nodes_root, inst_infos, labels):
            instances.append(InstanceNodeSup(node_root, inst_info, path_cost_sup=label))
        self.times.record_time("instances", time.time() - start_time)

        return instances


@pathfinding_factory.register_class("sup_q")
class PathFindEdgeSup(PathFindEdgeStatic[EdgesSupervisable, Any, InstanceEdgeSup], PathFindSup[EdgesSupervisable, InstanceEdgeSup]):
    @staticmethod
    def domain_type() -> Type[EdgesSupervisable]:
        return EdgesSupervisable

    @staticmethod
    def description() -> str:
        return "Labels edges"

    def step(self, verbose: bool = False) -> Tuple[List[Node], List[EdgeQ]]:
        edges: List[EdgeQ] = []
        for instance in self.instances:
            node_root: Node = instance.root_node
            edge: EdgeQ = EdgeQ(node_root, instance.action, instance.path_cost_sup)
            edges.append(edge)
            node_root.backup_val = instance.path_cost_sup
            instance.itr += 1
            instance.add_edges_popped([edge])
        start_time = time.time()
        # self.set_is_solved([edge.node for edge in edges])
        self.times.record_time("is_solved", time.time() - start_time)

        return [], edges

    def _compute_costs(self, instances: List[InstanceEdgeSup], edges_by_inst: List[List[EdgeQ]]) -> List[List[float]]:
        raise NotImplementedError

    def make_instances_sup(self, steps_gen: List[int], inst_infos: Optional[List[Any]]) -> List[InstanceEdgeSup]:
        # get edges and labels
        start_time = time.time()
        states_start, goals, actions_init, labels = self.domain.samp_edges_and_labels(steps_gen)
        self.times.record_time("get_sup", time.time() - start_time)

        # make root nodes
        nodes_root: List[Node] = self._create_root_nodes(states_start, goals)

        # make instances
        start_time = time.time()
        if inst_infos is None:
            inst_infos = [None for _ in states_start]

        instances: List[InstanceEdgeSup] = []
        for node_root, inst_info, action_init, label in zip(nodes_root, inst_infos, actions_init, labels, strict=True):
            instances.append(InstanceEdgeSup(node_root, inst_info, action=action_init, path_cost_sup=label))
        self.times.record_time("instances", time.time() - start_time)

        return instances


@pathfinding_factory.register_class("sup_p")
class PathFindEdgeSamp(PathFindEdgeStatic[EdgesSampleable, Any, InstanceEdgeSup], PathFindSup[EdgesSampleable, InstanceEdgeSup]):
    @staticmethod
    def domain_type() -> Type[EdgesSampleable]:
        return EdgesSampleable

    @staticmethod
    def description() -> str:
        return "Gets problem instances and first edge to take from start state"

    def step(self, verbose: bool = False) -> Tuple[List[Node], List[EdgeQ]]:
        edges: List[EdgeQ] = []
        for instance in self.instances:
            node_root: Node = instance.root_node
            edge: EdgeQ = EdgeQ(node_root, instance.action, instance.path_cost_sup)
            edges.append(edge)
            node_root.backup_val = instance.path_cost_sup
            instance.itr += 1
            instance.add_edges_popped([edge])
        start_time = time.time()
        self.times.record_time("is_solved", time.time() - start_time)

        return [], edges

    def _compute_costs(self, instances: List[InstanceEdgeSup], edges_by_inst: List[List[EdgeQ]]) -> List[List[float]]:
        raise NotImplementedError

    def make_instances_sup(self, steps_gen: List[int], inst_infos: Optional[List[Any]]) -> List[InstanceEdgeSup]:
        # get edges and labels
        start_time = time.time()
        states_start, goals, actions_init = self.domain.samp_edges(steps_gen)
        self.times.record_time("get_sup", time.time() - start_time)

        # make root nodes
        nodes_root: List[Node] = self._create_root_nodes(states_start, goals)

        # make instances
        start_time = time.time()
        if inst_infos is None:
            inst_infos = [None for _ in states_start]

        instances: List[InstanceEdgeSup] = []
        for node_root, action_init, inst_info in zip(nodes_root, actions_init, inst_infos, strict=True):
            instances.append(InstanceEdgeSup(node_root, action_init, 0.0, inst_info))
        self.times.record_time("instances", time.time() - start_time)

        return instances

import random
from abc import ABC
from typing import List, Tuple, Dict, Optional, Any, TypeVar
from deepxube.base.env import Env, EnvEnumerableActs, State, Goal, Action
from deepxube.base.pathfinding import Instance, NodeQ, PathFindQ, NodeQAct
from deepxube.utils import misc_utils
from heapq import heappush, heappop, heapify
import numpy as np
import time


OpenSetElem = Tuple[float, int, NodeQAct]


class InstanceBWQS(Instance[NodeQ]):
    def __init__(self, root_node: NodeQ, batch_size: int, weight: float, eps: float, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.open_set: List[OpenSetElem] = []
        self.heappush_count: int = 0
        self.closed_dict: Dict[State, float] = {}
        self.ub: float = np.inf
        self.lb: float = 0.0
        self.batch_size: int = batch_size
        self.weight: float = weight
        self.eps: float = eps

        self.push_to_open([NodeQAct(self.root_node, None, self.root_node.heuristic)], [self.root_node.heuristic])

    def push_to_open(self, nodeacts: List[NodeQAct], costs: List[float]) -> None:
        for nodeact, cost in zip(nodeacts, costs, strict=True):
            heappush(self.open_set, (cost, self.heappush_count, nodeact))
            self.heappush_count += 1

    def check_closed(self, nodes: List[NodeQ]) -> List[NodeQ]:
        nodes_ret: List[NodeQ] = []
        for node in nodes:
            path_cost_prev: Optional[float] = self.closed_dict.get(node.state)
            if (path_cost_prev is None) or (path_cost_prev > node.path_cost):
                self.closed_dict[node.state] = node.path_cost
                nodes_ret.append(node)
        return nodes_ret

    def pop_from_open(self) -> List[NodeQAct]:
        num_to_pop: int = min(self.batch_size, len(self.open_set))

        elems_popped: List[OpenSetElem] = []
        for _ in range(num_to_pop):
            if random.random() < self.eps:
                pop_idx: int = random.randrange(0, len(self.open_set))
                elems_popped.append(self.open_set.pop(pop_idx))
                heapify(self.open_set)
            else:
                elems_popped.append(heappop(self.open_set))
        nodeacts_popped: List[NodeQAct] = [elem_popped[2] for elem_popped in elems_popped]

        if len(elems_popped) > 0:
            cost_first: float = elems_popped[0][0]
            self.lb = max(cost_first, self.lb)

        return nodeacts_popped

    def update_ub(self, nodes: List[NodeQ]) -> None:
        # keep solved nodes for training
        for node in nodes:
            assert node.is_solved is not None
            if node.is_solved and (self.ub > node.path_cost):
                self.goal_node = node
                self.ub = node.path_cost

    def finished(self) -> bool:
        return (self.goal_node is not None) and (self.lb >= (self.weight * self.ub))


E = TypeVar('E', bound=Env)


class BWQS(PathFindQ[E, InstanceBWQS], ABC):
    def step(self, verbose: bool = False) -> List[NodeQAct]:
        # split instances by iteration
        instances_all: List[InstanceBWQS] = [instance for instance in self.instances if not instance.finished()]
        instances_itr0: List[InstanceBWQS] = [instance for instance in instances_all if instance.itr == 0]
        instances_itrgt0: List[InstanceBWQS] = [instance for instance in instances_all if instance.itr > 0]

        # pop from open
        start_time = time.time()
        nodeacts_popped_itr0: List[List[NodeQAct]] = [instance.pop_from_open() for instance in instances_itr0]
        nodeacts_popped_itrgt0: List[List[NodeQAct]] = [instance.pop_from_open() for instance in instances_itrgt0]
        self.times.record_time("pop", time.time() - start_time)

        # next state
        nodes_next_itr0: List[List[NodeQ]] = []
        for nodeacts_popped_itr0_i in nodeacts_popped_itr0:
            for nodeact in nodeacts_popped_itr0_i:
                assert nodeact.action is None
            nodes_next_itr0.append([nodeact.node for nodeact in nodeacts_popped_itr0_i])
        nodes_next_itrgt0: List[List[NodeQ]] = self.get_next_nodes(instances_itrgt0, nodeacts_popped_itrgt0)
        instances: List[InstanceBWQS] = instances_itr0 + instances_itrgt0
        nodes_next_by_inst: List[List[NodeQ]] = nodes_next_itr0 + nodes_next_itrgt0
        # nodes_next_flat: List[NodeQ] = misc_utils.flatten(nodes_next_by_inst)[0]

        # ub
        start_time = time.time()
        for instance, nodes_next in zip(instances, nodes_next_by_inst, strict=True):
            instance.update_ub(nodes_next)
        self.times.record_time("ub", time.time() - start_time)

        # check closed
        start_time = time.time()
        for inst_idx, instance in enumerate(instances):
            nodes_next_by_inst[inst_idx] = instance.check_closed(nodes_next_by_inst[inst_idx])
        self.times.record_time("check", time.time() - start_time)

        start_time = time.time()
        nodeacts_next_by_inst: List[List[NodeQAct]] = []

        for nodes_next in nodes_next_by_inst:
            nodeacts_next: List[NodeQAct] = []
            for node in nodes_next:
                for action, q_val in zip(node.actions, node.q_values, strict=True):
                    nodeacts_next.append(NodeQAct(node, action, q_val))
            nodeacts_next_by_inst.append(nodeacts_next)
        self.times.record_time("nodeacts", time.time() - start_time)

        # costs
        start_time = time.time()
        nodeacts_next_flat: List[NodeQAct] = misc_utils.flatten(nodeacts_next_by_inst)[0]
        weights, split_idxs = misc_utils.flatten([[instance.weight] * len(nodeacts_next)
                                                  for instance, nodeacts_next in
                                                  zip(instances, nodeacts_next_by_inst, strict=True)])
        path_costs: List[float] = [nodeact.node.path_cost for nodeact in nodeacts_next_flat]
        heuristics: List[float] = [nodeact.q_val for nodeact in nodeacts_next_flat]
        costs_flat: List[float] = ((np.array(weights) * np.array(path_costs)) + np.array(heuristics)).tolist()
        costs_by_inst: List[List[float]] = misc_utils.unflatten(costs_flat, split_idxs)
        self.times.record_time("cost", time.time() - start_time)

        # push to open
        start_time = time.time()
        for instance, nodeacts_next, costs in zip(instances, nodeacts_next_by_inst, costs_by_inst, strict=True):
            instance.push_to_open(nodeacts_next, costs)
        self.times.record_time("push", time.time() - start_time)

        # verbose
        if verbose:
            if len(heuristics) > 0:
                min_heur = float(np.min(heuristics))
                min_heur_pc = float(path_costs[np.argmin(heuristics)])
                max_heur = float(np.max(heuristics))
                max_heur_pc = float(path_costs[np.argmax(heuristics)])
                per_has_soln: float = 100.0 * float(np.mean([inst.has_soln() for inst in instances]))
                per_finished: float = 100.0 * float(np.mean([inst.finished for inst in instances]))

                print(f"Itr: %i, Added to OPEN - Min/Max Heur(PathCost): "
                      f"%.2f(%.2f)/%.2f(%.2f), %%has_soln: {per_has_soln}, "
                      f"%%finished: {per_finished}" % (self.itr, min_heur, min_heur_pc, max_heur, max_heur_pc))

            print(f"Times - {self.times.get_time_str()}")
            print("")

        # update iterations
        self.itr += 1
        for instance in instances:
            instance.itr += 1

        # return
        nodeacts_popped_by_inst: List[List[NodeQAct]] = nodeacts_popped_itr0 + nodeacts_popped_itrgt0
        nodesacts_popped_flat: List[NodeQAct] = misc_utils.flatten(nodeacts_popped_by_inst)[0]
        # nodes_popped_flat: List[NodeQ] = [nodeact_popped.node for nodeact_popped in nodesacts_popped_flat]
        return nodesacts_popped_flat


class BWQSEnum(BWQS[EnvEnumerableActs]):
    def get_qvals_acts(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[float]], List[List[Action]]]:
        actions_l: List[List[Action]] = self.env.get_state_actions(states)
        qvals_l: List[List[float]] = self.heur_fn(states, goals, actions_l)
        return qvals_l, actions_l

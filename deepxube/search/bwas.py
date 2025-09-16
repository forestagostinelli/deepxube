from typing import List, Tuple, Dict, Optional, Any
from deepxube.environments.environment_abstract import Environment, State, Goal
from deepxube.search.search_abstract import Search, Node, Instance
from deepxube.nnet.nnet_utils import HeurFN_T
import numpy as np
from heapq import heappush, heappop

from deepxube.utils import misc_utils
import time
from numpy.typing import NDArray


OpenSetElem = Tuple[float, int, Node]


class InstanceBWAS(Instance):
    def __init__(self, root_node: Node, inst_info: Any, weight: float):
        super().__init__(root_node, inst_info)
        self.open_set: List[OpenSetElem] = []
        self.heappush_count: int = 0
        self.closed_dict: Dict[State, float] = dict()
        self.finished: bool = False
        self.weight: float = weight

        self.check_and_push([self.root_node], [self.root_node.heuristic])

    def check_and_push(self, nodes: List[Node], costs: List[float]):
        assert len(nodes) == len(costs), "should have same length"
        for node, cost in zip(nodes, costs):
            # check
            path_cost_prev: Optional[float] = self.closed_dict.get(node.state)
            if (path_cost_prev is None) or (path_cost_prev > node.path_cost):
                self.closed_dict[node.state] = node.path_cost
                heappush(self.open_set, (cost, self.heappush_count, node))
                self.heappush_count += 1

    def pop_from_open(self, num_nodes: int) -> List[Node]:
        num_to_pop: int = min(num_nodes, len(self.open_set))

        elems_popped: List[OpenSetElem] = [heappop(self.open_set) for _ in range(num_to_pop)]
        nodes_popped: List[Node] = [elem_popped[2] for elem_popped in elems_popped]

        for node in nodes_popped:
            if node.is_solved and ((self.goal_node is None) or (node.path_cost < self.goal_node.path_cost)):
                self.goal_node = node

        # TODO check if elems_popped len is 0
        cost_first: float = elems_popped[0][0]
        if (self.goal_node is not None) and ((self.weight * self.goal_node.path_cost) <= cost_first):
            self.finished = True

        return nodes_popped


class BWAS(Search[InstanceBWAS]):
    def __init__(self, env: Environment):
        super().__init__(env)
        self.steps: int = 0

    def add_instances(self, states: List[State], goals: List[Goal], heur_fn: HeurFN_T,
                      inst_infos: Optional[List[Any]] = None, compute_init_heur: bool = True,
                      weights: Optional[List[float]] = None):
        start_time = time.time()
        if inst_infos is None:
            inst_infos = [None] * len(states)
        if weights is None:
            weights = [1.0] * len(states)

        assert len(states) == len(goals) == len(inst_infos) == len(weights), "Number should be the same"
        # compute starting costs
        if compute_init_heur:
            heuristics: NDArray = heur_fn(states, goals)
        else:
            heuristics: NDArray = np.zeros(len(states)).astype(np.float64)

        root_nodes: List[Node] = []
        is_solved_states: NDArray[np.bool_] = np.array(self.env.is_solved(states, goals))
        for state, goal, heuristic, is_solved in zip(states, goals, heuristics, is_solved_states):
            root_node: Node = Node(state, goal, 0.0, heuristic, is_solved, None, None, None)
            root_nodes.append(root_node)

        # initialize instances
        for root_node, inst_info, weight in zip(root_nodes, inst_infos, weights):
            self.instances.append(InstanceBWAS(root_node, inst_info, weight))
        self.times.record_time("add", time.time() - start_time)

    def step(self, heur_fn: HeurFN_T, batch_size: int = 1,
             verbose: bool = False) -> Tuple[List[State], List[Goal], List[float]]:
        instances: List[InstanceBWAS] = [instance for instance in self.instances if not instance.finished]

        # Pop from open
        start_time = time.time()
        nodes_by_inst_popped: List[List[Node]] = [instance.pop_from_open(batch_size) for instance in instances]
        self.times.record_time("pop", time.time() - start_time)

        # Expand nodes
        nodes_c_by_inst: List[List[Node]] = self.expand_nodes(instances, nodes_by_inst_popped, heur_fn)

        # Get cost
        start_time = time.time()
        nodes_c_flat, _ = misc_utils.flatten(nodes_c_by_inst)
        weights, split_idxs = misc_utils.flatten([[instance.weight] * len(nodes_c)
                                                  for instance, nodes_c in zip(instances, nodes_c_by_inst)])
        path_costs: List[float] = [node.path_cost for node in nodes_c_flat]
        heuristics: List[float] = [node.heuristic for node in nodes_c_flat]
        costs_flat: List[float] = list((np.array(weights) * np.array(path_costs)) + np.array(heuristics))
        costs_by_inst: List[List[float]] = misc_utils.unflatten(costs_flat, split_idxs)
        self.times.record_time("cost", time.time() - start_time)

        # Check if children are in closed and push if not
        start_time = time.time()
        assert len(instances) == len(nodes_c_by_inst) == len(costs_by_inst)
        for instance, nodes_c, costs in zip(instances, nodes_c_by_inst, costs_by_inst):
            assert len(nodes_c) == len(costs)
            instance.check_and_push(nodes_c, costs)
        self.times.record_time("check_push", time.time() - start_time)

        # Print to screen
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
                      f"%%finished: {per_finished}" % (self.steps, min_heur, min_heur_pc, max_heur, max_heur_pc))

            print(f"Times - {self.times.get_time_str()}")
            print("")

        # update iterations
        self.steps += 1
        for instance in instances:
            instance.itr += 1

        # return
        start_time = time.time()
        nodes_popped_flat, _ = misc_utils.flatten(nodes_by_inst_popped)
        states: List[State] = [node.state for node in nodes_popped_flat]
        goals: List[Goal] = [node.goal for node in nodes_popped_flat]
        ctgs: List[float] = [node.bellman_backup() for node in nodes_popped_flat]
        self.times.record_time("bellman", time.time() - start_time)
        return states, goals, ctgs

    def remove_finished_instances(self, itr_max: int) -> List[InstanceBWAS]:
        def remove_instance_fn(inst_in: InstanceBWAS) -> bool:
            if inst_in.finished:
                return True
            if inst_in.itr >= itr_max:
                return True
            return False

        return self.remove_instances(remove_instance_fn)

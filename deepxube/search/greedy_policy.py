from typing import List, Optional, Any, Tuple
from deepxube.environments.environment_abstract import State, Goal
from deepxube.nnet.nnet_utils import HeurFN_T
from deepxube.search.search_abstract import Node, Instance, Search
import numpy as np
from numpy.typing import NDArray
import random
import time


class InstanceGr(Instance):
    def __init__(self, root_node: Node, inst_info: Any, eps: float):
        super().__init__(root_node, inst_info)
        self.curr_node: Node = self.root_node
        self.eps = eps


class Greedy(Search[InstanceGr]):
    def add_instances(self, states: List[State], goals: List[Goal], heur_fn: HeurFN_T,
                      inst_infos: Optional[List[Any]] = None, compute_init_heur: bool = True,
                      eps_l: Optional[List[float]] = None):
        start_time = time.time()
        if inst_infos is None:
            inst_infos = [None] * len(states)
        if eps_l is None:
            eps_l = [0.0] * len(states)

        assert len(states) == len(goals) == len(inst_infos) == len(eps_l), "Number should be the same"

        if compute_init_heur:
            heuristics: NDArray = heur_fn(states, goals)
        else:
            heuristics: NDArray = np.zeros(len(states)).astype(np.float64)

        is_solved_l: List[bool] = self.env.is_solved(states, goals)

        for state, goal, heuristic, is_solved, inst_info, eps_inst in zip(states, goals, heuristics, is_solved_l,
                                                                          inst_infos, eps_l):
            root_node: Node = Node(state, goal, 0.0, heuristic, is_solved, None, None, None)
            instance: InstanceGr = InstanceGr(root_node, inst_info, eps_inst)
            self.instances.append(instance)
        self.times.record_time("add", time.time() - start_time)

    def step(self, heur_fn: HeurFN_T) -> Tuple[List[State], List[Goal], List[float]]:
        # get unsolved instances
        instances: List[InstanceGr] = self._get_unsolved_instances()
        if len(instances) == 0:
            return [], [], []

        self.expand_nodes(instances, [[inst.curr_node] for inst in instances], heur_fn)
        start_time = time.time()
        states: List[State] = [inst.curr_node.state for inst in instances]
        goals: List[Goal] = [inst.curr_node.goal for inst in instances]
        ctgs: List[float] = [inst.curr_node.bellman_backup() for inst in instances]
        self.times.record_time("bellman", time.time() - start_time)

        # take action
        start_time = time.time()
        rand_vals = np.random.random(len(instances))
        for idx, instance in enumerate(instances):
            # check solved
            if instance.curr_node.is_solved:
                instance.goal_node = instance.curr_node
            else:
                # get next state
                curr_node: Node = instance.curr_node
                assert curr_node.children is not None
                assert curr_node.t_costs is not None
                t_costs: List[float] = curr_node.t_costs
                children: List[Node] = curr_node.children
                tc_p_ctg_next: List[float] = [t_cost + child.heuristic for t_cost, child in zip(t_costs, children)]

                child_idx: int = int(np.argmin(tc_p_ctg_next))
                if rand_vals[idx] < instance.eps:
                    child_idx: int = random.choice(list(range(len(tc_p_ctg_next))))
                node_next: Node = children[child_idx]

                instance.curr_node = node_next
            instance.itr += 1
        self.times.record_time("get_next", time.time() - start_time)

        return states, goals, ctgs

    def remove_finished_instances(self, itr_max: int) -> List[InstanceGr]:
        def remove_instance_fn(inst_in: InstanceGr) -> bool:
            if inst_in.has_soln():
                return True
            if inst_in.itr >= itr_max:
                return True
            return False

        return self.remove_instances(remove_instance_fn)

    def _get_unsolved_instances(self) -> List[InstanceGr]:
        instances_unsolved: List[InstanceGr] = [instance for instance in self.instances if not instance.has_soln()]
        return instances_unsolved

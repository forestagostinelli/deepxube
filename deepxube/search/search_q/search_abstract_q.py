import time
from abc import abstractmethod
from typing import List, Optional, Tuple, TypeVar, Callable

import numpy as np
from numpy.typing import NDArray

from deepxube.search.search_abstract import Search, Node, Instance
from deepxube.environments.environment_abstract import Environment, State, Goal, Action


class NodeQ(Node):
    __slots__ = ['state', 'goal', 'path_cost', 'heuristic', 'is_solved', 'parent_action', 'parent_t_cost', 'parent',
                 'children', 't_costs', 'bellman_backup_val']
    def __init__(self, state: State, goal: Goal, path_cost: float, heuristic: float, is_solved: bool,
                 parent_action: Optional[Action], parent_t_cost: Optional[float], parent: Optional['NodeQ']):
        super().__init__(state, goal, path_cost, heuristic, is_solved, parent_action, parent_t_cost, parent)
        self.parent: Optional[NodeQ] = parent
        self.children: Optional[List[NodeQ]] = None
        self.t_costs: Optional[List[float]] = None
        self.bellman_backup_val: Optional[float] = None

    def backup(self) -> float:
        self.bellman_backup_val: float
        if self.is_solved:
            self.bellman_backup_val = 0.0
        else:
            assert self.children is not None
            if len(self.children) == 0:
                self.bellman_backup_val = self.heuristic
            else:
                assert self.t_costs is not None

                self.bellman_backup_val = np.inf
                for node_c, tc in zip(self.children, self.t_costs):
                    self.bellman_backup_val = min(self.bellman_backup_val, tc + node_c.heuristic)

        return self.bellman_backup_val

    def upper_bound_parent_path(self, ctg_ub: float):
        self.bellman_backup_val = min(self.bellman_backup_val, ctg_ub)
        if self.parent is not None:
            assert self.parent_t_cost is not None
            self.parent.upper_bound_parent_path(ctg_ub + self.parent_t_cost)


I = TypeVar('I', bound=Instance)


class SearchQ(Search[I]):
    def __init__(self, env: Environment):
        super().__init__(env)

    @abstractmethod
    def step(self, heur_fn: Callable) -> Tuple[List[State], List[Goal], List[Action], List[float]]:
        pass

    def _create_root_nodes(self, states: List[State], goals: List[Goal], heur_fn: Callable,
                           compute_init_heur: bool) -> List[NodeQ]:
        if compute_init_heur:
            heuristics: NDArray = heur_fn(states, goals)
        else:
            heuristics: NDArray = np.zeros(len(states)).astype(np.float64)

        root_nodes: List[NodeQ] = []
        is_solved_l: List[bool] = self.env.is_solved(states, goals)
        for state, goal, heuristic, is_solved in zip(states, goals, heuristics, is_solved_l):
            root_node: NodeQ = NodeQ(state, goal, 0.0, heuristic, is_solved, None, None, None)
            root_nodes.append(root_node)

        return root_nodes

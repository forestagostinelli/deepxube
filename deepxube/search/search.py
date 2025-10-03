from typing import Generic, List, Optional, Any, Tuple, Callable, TypeVar
from deepxube.environments.environment_abstract import Environment, State, Goal, Action
from deepxube.utils.timing_utils import Times

from abc import ABC, abstractmethod
import numpy as np


class Node(ABC):
    def __init__(self, state: State, goal: Goal, path_cost: float, heuristic: float, is_solved: bool,
                 parent_action: Optional[Action], parent_t_cost: Optional[float], parent: Optional['Node']):
        self.state: State = state
        self.goal: Goal = goal
        self.path_cost: float = path_cost
        self.heuristic: float = heuristic
        self.is_solved: bool = is_solved
        self.parent_action: Optional[Action] = parent_action
        self.parent_t_cost: Optional[float] = parent_t_cost
        self.parent: Optional[Node] = parent

    @abstractmethod
    def backup(self) -> float:
        pass


class Instance(ABC):
    def __init__(self, root_node: Node, inst_info: Any):
        self.root_node: Node = root_node
        self.itr: int = 0  # update with every search iteration
        self.num_nodes_generated: int = 0
        self.inst_info: Any = inst_info
        self.goal_node: Optional[Node] = None

    def has_soln(self) -> bool:
        if self.goal_node is None:
            return False
        else:
            return True

    def path_cost(self) -> float:
        if not self.has_soln():
            return np.inf
        else:
            assert self.goal_node is not None
            return self.goal_node.path_cost


I = TypeVar('I', bound=Instance)


class Search(ABC, Generic[I]):
    def __init__(self, env: Environment):
        self.env: Environment = env
        self.instances: List[I] = []
        self.times: Times = Times()

    @abstractmethod
    def add_instances(self, states: List[State], goals: List[Goal], heur_fn: Callable,
                      inst_infos: Optional[List[Any]] = None, compute_init_heur: bool = True, **kwargs):
        pass

    @abstractmethod
    def step(self, heur_fn: Callable) -> Any:
        pass

    @abstractmethod
    def remove_finished_instances(self, itr_max: int) -> List[I]:
        pass

    def remove_instances(self, test_rem: Callable[[I], bool]) -> List[I]:
        """ Remove instances

        :param test_rem: A Callable that takes an instance as input and returns true if the instance should be removed
        :return: List of removed instances
        """
        instances_remove: List[I] = []
        instances_keep: List[I] = []
        for instance in self.instances:
            if test_rem(instance):
                instances_remove.append(instance)
            else:
                instances_keep.append(instance)

        self.instances = instances_keep

        return instances_remove


def get_path(node: Node) -> Tuple[List[State], List[Action], float]:
    """ Gets path from the start state to the goal state associated with the input node

    :param node: goal node
    :return: List of states along path, List of actions in path, path cost
    """
    path: List[State] = []
    actions: List[Action] = []

    parent_node: Node = node
    while parent_node.parent is not None:
        path.append(parent_node.state)

        assert parent_node.parent_action is not None, "parent_action should not be None"
        actions.append(parent_node.parent_action)
        parent_node = parent_node.parent

    path.append(parent_node.state)

    path = path[::-1]
    actions = actions[::-1]

    return path, actions, node.path_cost

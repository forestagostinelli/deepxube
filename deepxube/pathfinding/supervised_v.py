from typing import List, Any, Optional, Type
from deepxube.base.domain import Domain, State, Goal, Action, StartGoalWalkable
from deepxube.base.pathfinding import Instance, Node, PathFind


class InstanceSupV(Instance):
    def __init__(self, root_node: Node, step_num: int, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.step_num: int = step_num

    def finished(self) -> bool:
        return self.itr > 0


class RWSupervisedV(PathFind[Domain, InstanceSupV]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    def step(self, verbose: bool = False) -> List[Node]:
        nodes: List[Node] = []
        for instance in self.instances:
            root_node: Node = instance.root_node
            root_node.heuristic = instance.step_num
            nodes.append(root_node)
            instance.itr += 1
        self.set_is_solved(nodes)

        return nodes

    def make_instances_rw(self, steps_gen: List[int], inst_infos: Optional[List[Any]]):
        """ Make instances from a random walk

        """
        if inst_infos is None:
            inst_infos = [None for _ in steps_gen]
        if isinstance(self.domain, StartGoalWalkable):
            states_start: List[State] = self.domain.get_start_states(len(steps_gen))
            states_goal: List[S] = self.random_walk(states_start, num_steps_l)[0]

    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: Optional[List[Any]] = None,
                       compute_root_heur: bool = True) -> List[InstanceSupV]:
        raise NotImplementedError

from typing import List, Any, Tuple
from deepxube.base.domain import Domain, State, Goal, Action
from deepxube.base.pathfinding import Instance, Node, PathFindV


class InstanceStepLenSup(Instance):
    def __init__(self, root_node: Node, step_num: int, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.step_num: int = step_num

    def finished(self) -> bool:
        return self.itr > 0


class StepLenSupV(PathFindV[Domain, InstanceStepLenSup]):
    def step(self, verbose: bool = False) -> List[Node]:
        nodes: List[Node] = []
        for instance in self.instances:
            root_node: Node = instance.root_node
            root_node.heuristic = instance.step_num
            nodes.append(root_node)
            instance.itr += 1
        self.set_is_solved(nodes)

        return nodes

    def _expand(self, states: List[State],
                goals: List[Goal]) -> Tuple[List[List[State]], List[List[Action]], List[List[float]]]:
        raise NotImplementedError

    def make_instances(self, states: List[State], goals: List[Goal], inst_infos: List[Any], compute_root_heur: bool) -> List[InstanceStepLenSup]:
        raise NotImplementedError

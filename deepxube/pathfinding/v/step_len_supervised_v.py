from typing import List, Any, Tuple
from deepxube.base.domain import Domain, State, Goal, Action
from deepxube.base.pathfinding import Instance, NodeV, PathFindV


class InstanceStepLenSup(Instance[NodeV]):
    def __init__(self, root_node: NodeV, step_num: int, inst_info: Any):
        super().__init__(root_node, inst_info)
        self.step_num: int = step_num

    def finished(self) -> bool:
        return self.itr > 0


class StepLenSupV(PathFindV[Domain, InstanceStepLenSup]):
    def step(self, verbose: bool = False) -> List[NodeV]:
        nodes: List[NodeV] = []
        for instance in self.instances:
            root_node: NodeV = instance.root_node
            root_node.children = [NodeV(root_node.state, root_node.goal, 0.0, instance.step_num, None, None, 0.0,
                                        root_node)]
            root_node.t_costs = [0.0]
            nodes.append(root_node)
            instance.itr += 1
        self.set_is_solved(nodes)

        return nodes

    def expand(self, states: List[State],
               goals: List[Goal]) -> Tuple[List[List[State]], List[List[Action]], List[List[float]]]:
        raise NotImplementedError

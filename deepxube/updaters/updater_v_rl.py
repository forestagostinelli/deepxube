import time
from typing import Dict, Any, List

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import Domain, State, Goal
from deepxube.base.heuristic import HeurNNetParV, HeurFnV
from deepxube.base.pathfinding import PathFindVHeur, Node, InstanceV
from deepxube.base.updater import UpdateHeurV, UpdateHeurRL, UpArgs, UpHeurArgs, P
from deepxube.utils.timing_utils import Times


def _pathfind_v_step(pathfind: PathFindVHeur) -> List[Node]:
    # take a step
    nodes_popped: List[Node] = pathfind.step()
    assert len(nodes_popped) == len(pathfind.instances), f"Values were {len(nodes_popped)} and {len(pathfind.instances)}"

    return nodes_popped


class UpdateHeurVRL(UpdateHeurV[Domain, PathFindVHeur], UpdateHeurRL[Domain, PathFindVHeur, InstanceV, HeurNNetParV, HeurFnV]):
    def __init__(self, domain: Domain, pathfind_name: str, pathfind_kwargs: Dict[str, Any], up_args: UpArgs, up_heur_args: UpHeurArgs):
        super().__init__(domain, pathfind_name, pathfind_kwargs, up_args)
        self.up_heur_args: UpHeurArgs = up_heur_args

    def get_up_args_repr(self) -> str:
        return f"{super().get_up_args_repr()}\n{self.up_heur_args.__repr__()}"

    def _step(self, pathfind: PathFindVHeur, times: Times) -> None:
        _pathfind_v_step(pathfind)

    def _step_sync_main(self, pathfind: P, times: Times) -> List[NDArray]:
        raise NotImplementedError

    def _get_instance_data(self, instances: List[InstanceV], times: Times) -> List[NDArray]:
        # get backup
        start_time = time.time()
        ctgs_backup: List[float] = []
        nodes_popped: List[Node] = []
        for instance in instances:
            nodes_popped.extend(instance.nodes_popped)
        if self.up_heur_args.backup == 1:
            for node in nodes_popped:
                node.bellman_backup()
            if self.up_heur_args.ub_heur_solns:
                for node in nodes_popped:
                    assert node.is_solved is not None
                    if node.is_solved:
                        node.upper_bound_parent_path(0.0)
        elif self.up_heur_args.backup == -1:
            for instance in instances:
                instance.root_node.tree_backup()
        else:
            raise ValueError(f"Unknown backup {self.up_heur_args.backup}")

        for node in nodes_popped:
            ctgs_backup.append(node.backup_val)
        times.record_time("backup", time.time() - start_time)

        start_time = time.time()
        states: List[State] = [node.state for node in nodes_popped]
        goals: List[Goal] = [node.goal for node in nodes_popped]
        inputs_np: List[NDArray] = self.get_heur_nnet_par().to_np(states, goals)
        times.record_time("to_np", time.time() - start_time)

        return inputs_np + [np.array(ctgs_backup)]

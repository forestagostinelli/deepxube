import time
from typing import Dict, Any, List

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import Domain, State, Goal
from deepxube.base.heuristic import HeurNNetParV, HeurFnV
from deepxube.base.pathfinding import PathFindVHeur, Node, Instance
from deepxube.base.updater import UpdateHeurV, UpdateHeurRL, UpArgs, UpHeurArgs
from deepxube.utils.timing_utils import Times


class UpdateHeurVRL(UpdateHeurV[Domain, PathFindVHeur], UpdateHeurRL[Domain, PathFindVHeur, HeurNNetParV, HeurFnV]):
    def __init__(self, domain: Domain, pathfind_name: str, pathfind_kwargs: Dict[str, Any], up_args: UpArgs, up_heur_args: UpHeurArgs):
        super().__init__(domain, pathfind_name, pathfind_kwargs, up_args)
        self.up_heur_args: UpHeurArgs = up_heur_args

    def _step(self, pathfind: PathFindVHeur, times: Times) -> List[NDArray]:
        # take a step
        nodes_popped: List[Node] = pathfind.step()
        assert len(nodes_popped) == len(pathfind.instances), f"Values were {len(nodes_popped)} and {len(pathfind.instances)}"
        if not self.up_args.sync_main:
            self.nodes_popped.extend(nodes_popped)
            return []
        else:
            # TODO implement for sync_main
            raise NotImplementedError

    def _get_instance_data(self, instances: List[Instance], times: Times) -> List[NDArray]:
        # get backup
        start_time = time.time()
        ctgs_backup: List[float] = []
        if self.up_heur_args.backup == 1:
            for node in self.nodes_popped:
                node.bellman_backup()
            if self.up_heur_args.ub_heur_solns:
                for node in self.nodes_popped:
                    assert node.is_solved is not None
                    if node.is_solved:
                        node.upper_bound_parent_path(0.0)
        elif self.up_heur_args.backup == -1:
            for instance in instances:
                root_node: Node = instance.root_node
                root_node.tree_backup()
        else:
            raise ValueError(f"Unknown backup {self.up_heur_args.backup}")

        for node in self.nodes_popped:
            ctgs_backup.append(node.backup_val)
        times.record_time("backup", time.time() - start_time)

        start_time = time.time()
        states: List[State] = [node.state for node in self.nodes_popped]
        goals: List[Goal] = [node.goal for node in self.nodes_popped]
        inputs_np: List[NDArray] = self.get_heur_nnet_par().to_np(states, goals)
        times.record_time("to_np", time.time() - start_time)

        self.nodes_popped = []
        return inputs_np + [np.array(ctgs_backup)]

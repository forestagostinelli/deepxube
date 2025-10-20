from typing import List, Any

from deepxube.base.env import EnvEnumerableActs
from deepxube.base.heuristic import HeurNNetV, HeurNNetQ
from deepxube.base.pathfinding import NodeV, NodeQ
from deepxube.base.updater import UpdateHeurV, UpdateHeurQEnum, UpArgs
from deepxube.pathfinding.v.bwas import BWASEnum, InstanceBWAS
from deepxube.utils.timing_utils import Times
from deepxube.pathfinding.q.bwqs import BWQSEnum, InstanceBWQS


class UpdateHeurBWASEnum(UpdateHeurV[EnvEnumerableActs, InstanceBWAS, BWASEnum]):
    def __init__(self, env: EnvEnumerableActs, up_args: UpArgs, heur_nnet: HeurNNetV):
        super().__init__(env, up_args)
        self.set_heur_nnet(heur_nnet)

    def get_pathfind(self) -> BWASEnum:
        return BWASEnum(self.env, self.get_heur_fn())

    def _get_instances(self, pathfind: BWASEnum, steps_gen: List[int], inst_infos: List[Any],
                       times: Times) -> List[InstanceBWAS]:
        root_nodes: List[NodeV] = self._get_root_nodes(pathfind, steps_gen, times)
        return [InstanceBWAS(root_node, 1, 1.0, inst_info) for root_node, inst_info in
                zip(root_nodes, inst_infos, strict=True)]


class UpdateHeurBWQSEnum(UpdateHeurQEnum[InstanceBWQS, BWQSEnum]):
    def __init__(self, env: EnvEnumerableActs, up_args: UpArgs, heur_nnet: HeurNNetQ, eps: float):
        super().__init__(env, up_args)
        self.set_heur_nnet(heur_nnet)
        self.eps: float = eps

    def get_pathfind(self) -> BWQSEnum:
        return BWQSEnum(self.env, self.get_heur_fn())

    def _get_instances(self, pathfind: BWQSEnum, steps_gen: List[int], inst_infos: List[Any],
                       times: Times) -> List[InstanceBWQS]:
        root_nodes: List[NodeQ] = self._get_root_nodes(pathfind, steps_gen, times)
        return [InstanceBWQS(root_node, 1, 1.0, self.eps, inst_info) for root_node, inst_info in
                zip(root_nodes, inst_infos, strict=True)]

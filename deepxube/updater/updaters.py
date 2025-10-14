from typing import List, Any

from deepxube.base.pathfinding import NodeV, NodeQ
from deepxube.base.updater import UpdateHeurV, UpdateHeurQEnum
from deepxube.pathfinding.v.bwas import BWAS, InstanceBWAS
from deepxube.pathfinding.q.bwqs import BWQSEnum, InstanceBWQS


class UpdateHeurBWAS(UpdateHeurV[InstanceBWAS, BWAS]):
    def get_pathfind(self) -> BWAS:
        assert self.heur_fn is not None
        return BWAS(self.env, self.heur_fn)

    def _get_instances(self, root_nodes: List[NodeV], inst_infos: List[Any]) -> List[InstanceBWAS]:
        return [InstanceBWAS(root_node, 1, 1.0, inst_info) for root_node, inst_info in
                zip(root_nodes, inst_infos, strict=True)]


class UpdateHeurBWQSEnum(UpdateHeurQEnum[InstanceBWQS, BWQSEnum]):
    def get_pathfind(self) -> BWQSEnum:
        assert self.heur_fn is not None
        return BWQSEnum(self.env, self.heur_fn)

    def _get_instances(self, root_nodes: List[NodeQ], inst_infos: List[Any]) -> List[InstanceBWQS]:
        return [InstanceBWQS(root_node, 1, 1.0, inst_info) for root_node, inst_info in
                zip(root_nodes, inst_infos, strict=True)]

from typing import List, Any
from dataclasses import dataclass

from deepxube.base.domain import Domain, ActsEnum
from deepxube.base.heuristic import HeurNNetParV, HeurNNetParQ
from deepxube.base.pathfinding import NodeV, NodeQ
from deepxube.base.updater import UpdateHeurV, UpdateHeurQEnum, UpHeurArgs, UpArgs

from deepxube.pathfinding.v.step_len_supervised_v import StepLenSupV, InstanceStepLenSup
from deepxube.pathfinding.v.bwas import BWASEnum, InstanceBWAS
from deepxube.pathfinding.v.greedy_policy import InstanceGrPolV, GreedyPolicyVEnum
from deepxube.pathfinding.q.greedy_policy_q import InstanceGrPolQ, GreedyPolicyQEnum
from deepxube.utils.timing_utils import Times
from deepxube.pathfinding.q.bwqs import BWQSEnum, InstanceBWQS


# supervised
class UpdateHeurRWSupV(UpdateHeurV[Domain, InstanceStepLenSup, StepLenSupV]):
    def get_pathfind(self) -> StepLenSupV:
        return StepLenSupV(self.env, self.get_heur_fn())

    def _get_instances(self, pathfind: StepLenSupV, steps_gen: List[int], inst_infos: List[Any],
                       times: Times) -> List[InstanceStepLenSup]:
        root_nodes: List[NodeV] = self._get_root_nodes(pathfind, steps_gen, times)
        return [InstanceStepLenSup(root_node, step_gen, inst_info) for root_node, step_gen, inst_info in
                zip(root_nodes, steps_gen, inst_infos, strict=True)]


# graph search
@dataclass
class UpGraphSearchArgs:
    """ Arguments when doing graph search, such as A* and Q*
    :param weight: search weight
    :param eps: expand random with probability eps
    """
    weight: float
    eps: float


class UpdateHeurBWASEnum(UpdateHeurV[ActsEnum, InstanceBWAS, BWASEnum]):
    def __init__(self, env: ActsEnum, up_args: UpArgs, up_heur_args: UpHeurArgs,
                 up_graphsch_args: UpGraphSearchArgs, heur_nnet: HeurNNetParV):
        super().__init__(env, up_args, up_heur_args, heur_nnet)
        self.up_graphsch_args: UpGraphSearchArgs = up_graphsch_args

    def get_pathfind(self) -> BWASEnum:
        return BWASEnum(self.env, self.get_heur_fn())

    def _get_instances(self, pathfind: BWASEnum, steps_gen: List[int], inst_infos: List[Any],
                       times: Times) -> List[InstanceBWAS]:
        root_nodes: List[NodeV] = self._get_root_nodes(pathfind, steps_gen, times)
        return [InstanceBWAS(root_node, 1, self.up_graphsch_args.weight, self.up_graphsch_args.eps, inst_info)
                for root_node, inst_info in zip(root_nodes, inst_infos, strict=True)]

    def get_up_args_repr(self) -> str:
        return f"{super().get_up_args_repr()}\n{self.up_graphsch_args.__repr__()}"


class UpdateHeurBWQSEnum(UpdateHeurQEnum[InstanceBWQS, BWQSEnum]):
    def __init__(self, env: ActsEnum, up_args: UpArgs, up_heur_args: UpHeurArgs,
                 up_graphsch_args: UpGraphSearchArgs, heur_nnet: HeurNNetParQ):
        super().__init__(env, up_args, up_heur_args, heur_nnet)
        self.up_graphsch_args: UpGraphSearchArgs = up_graphsch_args

    def get_pathfind(self) -> BWQSEnum:
        return BWQSEnum(self.env, self.get_heur_fn())

    def _get_instances(self, pathfind: BWQSEnum, steps_gen: List[int], inst_infos: List[Any],
                       times: Times) -> List[InstanceBWQS]:
        root_nodes: List[NodeQ] = self._get_root_nodes(pathfind, steps_gen, times)

        return [InstanceBWQS(root_node, 1, self.up_graphsch_args.weight, self.up_graphsch_args.eps, inst_info)
                for root_node, inst_info in zip(root_nodes, inst_infos, strict=True)]

    def get_up_args_repr(self) -> str:
        return f"{super().get_up_args_repr()}\n{self.up_graphsch_args.__repr__()}"


# greedy policy
@dataclass
class UpGreedyPolicyArgs:
    """ Arguments when doing a greedy policy
    :param temp: temperature for Boltzmann exploration
    :param eps: select random with probability eps
    """
    temp: float  # TODO special case when temp=0.0
    eps: float


class UpdateHeurGrPolVEnum(UpdateHeurV[ActsEnum, InstanceGrPolV, GreedyPolicyVEnum]):
    def __init__(self, env: ActsEnum, up_args: UpArgs, up_heur_args: UpHeurArgs,
                 up_grpol_args: UpGreedyPolicyArgs, heur_nnet: HeurNNetParV):
        super().__init__(env, up_args, up_heur_args, heur_nnet)
        self.up_greedy_args: UpGreedyPolicyArgs = up_grpol_args

    def get_pathfind(self) -> GreedyPolicyVEnum:
        return GreedyPolicyVEnum(self.env, self.get_heur_fn())

    def _get_instances(self, pathfind: GreedyPolicyVEnum, steps_gen: List[int], inst_infos: List[Any],
                       times: Times) -> List[InstanceGrPolV]:
        root_nodes: List[NodeV] = self._get_root_nodes(pathfind, steps_gen, times)

        return [InstanceGrPolV(root_node, self.up_greedy_args.eps, inst_info) for root_node, inst_info in
                zip(root_nodes, inst_infos, strict=True)]

    def get_up_args_repr(self) -> str:
        return f"{super().get_up_args_repr()}\n{self.up_greedy_args.__repr__()}"


class UpdateHeurGrPolQEnum(UpdateHeurQEnum[InstanceGrPolQ, GreedyPolicyQEnum]):
    def __init__(self, env: ActsEnum, up_args: UpArgs, up_heur_args: UpHeurArgs,
                 up_greedy_args: UpGreedyPolicyArgs, heur_nnet: HeurNNetParQ):
        super().__init__(env, up_args, up_heur_args, heur_nnet)
        self.up_greedy_args: UpGreedyPolicyArgs = up_greedy_args

    def get_pathfind(self) -> GreedyPolicyQEnum:
        return GreedyPolicyQEnum(self.env, self.get_heur_fn())

    def _get_instances(self, pathfind: GreedyPolicyQEnum, steps_gen: List[int], inst_infos: List[Any],
                       times: Times) -> List[InstanceGrPolQ]:
        root_nodes: List[NodeQ] = self._get_root_nodes(pathfind, steps_gen, times)

        return [InstanceGrPolQ(root_node, self.up_greedy_args.temp, self.up_greedy_args.eps, inst_info)
                for root_node, inst_info in zip(root_nodes, inst_infos, strict=True)]

    def get_up_args_repr(self) -> str:
        return f"{super().get_up_args_repr()}\n{self.up_greedy_args.__repr__()}"

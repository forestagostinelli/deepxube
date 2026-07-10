from typing import Any, Tuple, Dict, TypeVar, Generic, Optional, cast, Type, List
from dataclasses import dataclass
from abc import abstractmethod, ABC

from deepxube.nnet.nnet_utils import NNetPar, NNetFn, ProcessedInput, Ctx, NNetParRunner, NNF
from deepxube.utils import misc_utils
from deepxube.base.domain import Domain, State, Action, Goal, ActsEnumFixed
from deepxube.base.nnet_input import NNetInput, StateGoalIn, StateGoalActFixIn, StateGoalActIn, PolicyNNetIn
from deepxube.base.heuristic import DeepXubeNNet, HeurNNet, PolicyNNet
from deepxube.base.nnet_fn import HeurFn, HeurVFn, HeurQFn, PolicyFn
from deepxube.factories.nnet_input_factory import get_nnet_input_t
from deepxube.factories.heuristic_factory import deepxube_nnet_factory

import numpy as np
from numpy.typing import NDArray


D = TypeVar('D', bound=Domain)
NNInP = TypeVar('NNInP', bound=NNetInput)
DXNNet = TypeVar('DXNNet', bound=DeepXubeNNet)


class DeepXubeNNetPar(NNetPar, Generic[NNetFn, Ctx, D, NNInP, DXNNet]):
    @staticmethod
    @abstractmethod
    def domain_type() -> Type[D]:
        pass

    @staticmethod
    @abstractmethod
    def nnet_input_type() -> Type[NNInP]:
        pass

    @staticmethod
    @abstractmethod
    def nnet_type() -> Type[DXNNet]:
        pass

    def __init__(self, domain: D, nnet_input_name: Tuple[str, str], nnet_name: str, nnet_kwargs: Dict[str, Any], **kwargs: Any):
        assert isinstance(domain, self.domain_type()), f"Domain {domain} must be an instance of {self.domain_type()}."
        nnet_input_t: Type[NNetInput] = get_nnet_input_t(nnet_input_name)
        assert issubclass(nnet_input_t, self.nnet_input_type()), (f"NNetInput {nnet_input_t} (name {nnet_input_name}) must be a subclass of "
                                                                  f"{self.nnet_input_type()}.")
        nnet_t: Type[DeepXubeNNet] = deepxube_nnet_factory.get_type(nnet_name)
        assert issubclass(nnet_t, self.nnet_type()), f"DeepXubeNNet {nnet_t} (name {nnet_name}) must be a subclass of {self.nnet_type()}."

        self.domain: D = domain
        self.nnet_input_name: Tuple[str, str] = nnet_input_name
        self.nnet_name: str = nnet_name
        self.nnet_kwargs: Dict[str, Any] = nnet_kwargs

        super().__init__(**kwargs)

        self.nnet_input: Optional[NNInP] = None

    def get_nnet(self) -> DXNNet:
        nnet_params: Dict = self.nnet_kwargs.copy()
        nnet_params['nnet_input'] = self._get_nnet_input()
        self._add_nnet_kwargs(nnet_params)
        nnet: DeepXubeNNet = deepxube_nnet_factory.build_class(self.nnet_name, nnet_params)
        assert isinstance(nnet, self.nnet_type())
        return nnet

    @abstractmethod
    def _add_nnet_kwargs(self, nnet_params: Dict) -> None:
        pass

    def _get_nnet_input(self) -> NNInP:
        if self.nnet_input is None:
            self.nnet_input = cast(NNInP, get_nnet_input_t(self.nnet_input_name)(domain=self.domain))

        assert self.nnet_input is not None
        return self.nnet_input

    def __getstate__(self) -> Dict:
        self.nnet_input = None
        return self.__dict__


H = TypeVar('H', bound=HeurFn)


class HeurNNetPar(DeepXubeNNetPar[H, Ctx, D, NNInP, HeurNNet], ABC):
    @staticmethod
    def nnet_type() -> Type[HeurNNet]:
        return HeurNNet

    @abstractmethod
    def _qfix(self) -> bool:
        pass

    @abstractmethod
    def _out_dim(self) -> int:
        pass

    def _add_nnet_kwargs(self, nnet_params: Dict) -> None:
        nnet_params["q_fix"] = self._qfix()
        nnet_params["out_dim"] = self._out_dim()


class HeurVNNetPar(HeurNNetPar[HeurVFn, None, Domain, StateGoalIn]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    @staticmethod
    def nnet_input_type() -> Type[StateGoalIn]:
        return StateGoalIn

    def process_inputs(self, states: List[State], goals: List[Goal]) -> ProcessedInput[None]:
        return ProcessedInput(self._get_nnet_input().to_np(states, goals), None)

    def process_outputs(self, outs: List[NDArray], update_num: Optional[int], ctx: None) -> List[float]:
        heurs: NDArray = outs[0]
        heurs = np.maximum(heurs[:, 0], 0)
        if (update_num is not None) and (update_num == 0):
            heurs = heurs * 0
        return cast(List[float], heurs.astype(np.float64).tolist())

    def _qfix(self) -> bool:
        return False

    def _out_dim(self) -> int:
        return 1


class HeurQNNetPar(HeurNNetPar[HeurQFn, Ctx, D, NNInP], ABC):
    @abstractmethod
    def process_inputs(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> ProcessedInput[Ctx]:
        pass


@dataclass(frozen=True)
class QOutFixCtx:
    states: List[State]


class HeurQNNetParFixOut(HeurQNNetPar[QOutFixCtx, ActsEnumFixed, StateGoalActFixIn]):
    @staticmethod
    def domain_type() -> Type[ActsEnumFixed]:
        return ActsEnumFixed

    @staticmethod
    def nnet_input_type() -> Type[StateGoalActFixIn]:
        return StateGoalActFixIn

    @staticmethod
    def _check_same_num_acts(actions_l: List[List[Action]]) -> None:
        assert len(set(len(actions) for actions in actions_l)) == 1, "num actions should be the same for all instances"

    def process_inputs(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> ProcessedInput[QOutFixCtx]:
        self._check_same_num_acts(actions_l)
        return ProcessedInput(self._get_nnet_input().to_np(states, goals, actions_l), QOutFixCtx(states))

    def process_outputs(self, outs: List[NDArray], update_num: Optional[int], ctx: QOutFixCtx) -> List[List[float]]:
        q_vals_np: NDArray = outs[0]
        assert q_vals_np.shape[0] == len(ctx.states)

        q_vals_np = np.maximum(q_vals_np, 0)
        if (update_num is not None) and (update_num == 0):
            q_vals_np = q_vals_np * 0
        q_vals_l: List[List[float]] = [q_vals_np[state_idx].astype(np.float64).tolist() for state_idx in range(q_vals_np.shape[0])]
        return q_vals_l

    def _qfix(self) -> bool:
        return True

    def _out_dim(self) -> int:
        return self.domain.get_num_acts()


@dataclass(frozen=True)
class QInCtx:
    states_rep: List[State]
    split_idxs: List[int]


class HeurQNNetParIn(HeurQNNetPar[QInCtx, Domain, StateGoalActIn]):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    @staticmethod
    def nnet_input_type() -> Type[StateGoalActIn]:
        return StateGoalActIn

    def process_inputs(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> ProcessedInput[QInCtx]:
        actions_flat, split_idxs = misc_utils.flatten(actions_l)
        states_rep: List[State] = []
        goals_rep: List[Goal] = []
        for state, goal, actions in zip(states, goals, actions_l, strict=True):
            states_rep.extend([state] * len(actions))
            goals_rep.extend([goal] * len(actions))

        return ProcessedInput(self._get_nnet_input().to_np(states_rep, goals_rep, actions_flat), QInCtx(states_rep, split_idxs))

    def process_outputs(self, outs: List[NDArray], update_num: Optional[int], ctx: QInCtx) -> List[List[float]]:
        q_vals_np: NDArray = outs[0]

        assert q_vals_np.shape[0] == len(ctx.states_rep)
        q_vals_np = np.maximum(q_vals_np[:, 0], 0)
        if (update_num is not None) and (update_num == 0):
            q_vals_np = q_vals_np * 0

        q_vals_flat: List[float] = q_vals_np.astype(np.float64).tolist()
        q_vals_l: List[List[float]] = misc_utils.unflatten(q_vals_flat, ctx.split_idxs)
        return q_vals_l

    def _qfix(self) -> bool:
        return False

    def _out_dim(self) -> int:
        return 1


@dataclass(frozen=True)
class PolicyCtx:
    num_states: int


class PolicyNNetPar(DeepXubeNNetPar[PolicyFn, PolicyCtx, Domain, PolicyNNetIn, PolicyNNet]):
    def __init__(self, *args: Any, num_samp: int = 0, **kwargs: Any):
        assert num_samp > 0
        self.num_samp: int = num_samp
        super().__init__(*args, **kwargs)

    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    @staticmethod
    def nnet_input_type() -> Type[PolicyNNetIn]:
        return PolicyNNetIn

    @staticmethod
    def nnet_type() -> Type[PolicyNNet]:
        return PolicyNNet

    def to_np_train(self, states: List[State], goals: List[Goal], actions: List[Action]) -> List[NDArray[Any]]:
        return self._get_nnet_input().to_np(states, goals, actions)

    def process_inputs(self, states: List[State], goals: List[Goal]) -> ProcessedInput[PolicyCtx]:
        return ProcessedInput(self._get_nnet_input().to_np_fn(states, goals), PolicyCtx(len(states)))

    def process_outputs(self, outs: List[NDArray], update_num: Optional[int], ctx: PolicyCtx) -> Tuple[List[List[Action]], List[List[float]]]:
        actions_np: List[NDArray] = outs[0:-1]
        pdfs_np: NDArray = outs[-1]

        # assert dimensions match
        assert len(pdfs_np.shape) == 2
        assert pdfs_np.shape[0] == ctx.num_states
        for actions_np_i in actions_np:
            assert actions_np_i.shape[0] == ctx.num_states
            assert actions_np_i.shape[0] == pdfs_np.shape[0]
            assert actions_np_i.shape[1] == pdfs_np.shape[1]

        # convert to action object rep
        actions_l: List[List[Action]] = []
        pdfs_l: List[List[float]] = []
        for state_idx in range(ctx.num_states):
            actions_np_state: List[NDArray[np.float64]] = [actions_np_i[state_idx] for actions_np_i in actions_np]
            pdfs_state: List[float] = pdfs_np[state_idx, :].tolist()

            actions_l.append(self._get_nnet_input().nnet_out_to_actions(actions_np_state))
            pdfs_l.append(pdfs_state)

        return actions_l, pdfs_l

    def _add_nnet_kwargs(self, nnet_params: Dict) -> None:
        nnet_params["num_samp"] = self.num_samp

    def __repr__(self) -> str:
        nnet: PolicyNNet = self.get_nnet()
        return f"{super().__repr__()}\n#Samp: {nnet.num_samp}"


class HeurVNNetParRunner(NNetParRunner[HeurVFn, HeurVNNetPar]):
    @staticmethod
    def nnet_fn_type() -> Type[HeurVFn]:
        return HeurVFn


class HeurQNNetParRunner(NNetParRunner[HeurQFn, HeurQNNetPar]):
    @staticmethod
    def nnet_fn_type() -> Type[HeurQFn]:
        return HeurQFn


class PolicyNNetParRunner(NNetParRunner[PolicyFn, PolicyNNetPar]):
    @staticmethod
    def nnet_fn_type() -> Type[PolicyFn]:
        return PolicyFn


from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, Union, runtime_checkable, Protocol, Tuple, TypeVar, Generic, Type, Dict, Any, Optional, cast

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import State, Action, Goal, Domain
from deepxube.base.heuristic import DeepXubeNNet, HeurNNet, PolicyNNet
from deepxube.base.nnet_input import NNetInput, StateGoalIn, PolicyNNetIn
from deepxube.factories.heuristic_factory import deepxube_nnet_factory
from deepxube.factories.nnet_input_factory import get_nnet_input_t
from deepxube.nnet.nnet_utils import NNetPar, NNF_T, CTX_T, ProcessedInput


# Individual functions

@runtime_checkable
class HeurVFn(Protocol):
    """ Maps states and goals to cost-to-go """
    def __call__(self, states: List[State], goals: List[Goal]) -> List[float]:
        ...


@runtime_checkable
class HeurQFn(Protocol):
    """ Maps states, goals, and actions to transitions cost plus cost-to-go of resulting state """
    def __call__(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> List[List[float]]:
        ...


HeurFn = Union[HeurVFn, HeurQFn]


@runtime_checkable
class PolicyFn(Protocol):
    """ Samples actions and their corresponding log probabilities given states and goals """
    def __call__(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[Action]], List[List[float]]]:
        """ Map states and goals to sampled actions along with their probability (or log probability) densities

        """
        ...


# Pathfind functions

@dataclass(frozen=True)
class PFNs:
    pass


@dataclass(frozen=True)
class PFNsHeurV(PFNs):
    heurv: HeurVFn


@dataclass(frozen=True)
class PFNsHeurQ(PFNs):
    heurq: HeurQFn


@dataclass(frozen=True)
class PFNsPolicy(PFNs):
    policy: PolicyFn


@dataclass(frozen=True)
class PFNsHeurVPolicy(PFNsPolicy, PFNsHeurV):
    pass


@dataclass(frozen=True)
class PFNsHeurQPolicy(PFNsPolicy, PFNsHeurQ):
    pass


# Parallel neural network functions

D = TypeVar('D', bound=Domain)
NNInP = TypeVar('NNInP', bound=NNetInput)
DXNNet = TypeVar('DXNNet', bound=DeepXubeNNet)


class DeepXubeNNetPar(NNetPar[NNF_T, CTX_T], Generic[NNF_T, CTX_T, D, NNInP, DXNNet]):
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

    @classmethod
    def get_incompat_reason(cls, domain: Domain, nnet_input_t: Type[NNetInput], nnet_t: Type[DeepXubeNNet]) -> Optional[str]:
        if not isinstance(domain, cls.domain_type()):
            return f"Domain {domain} is not an instance of {cls.domain_type()}"
        elif not issubclass(nnet_input_t, cls.nnet_input_type()):
            return f"NNetInput type {nnet_input_t} is not a subclass of {cls.nnet_input_type()}"
        elif not issubclass(nnet_t, cls.nnet_type()):
            return f"DeepXubeNNet type {nnet_t} is not a subclass of {cls.nnet_type()}"
        elif not issubclass(nnet_input_t, nnet_t.nnet_input_type()):
            return f"NNetInput type {nnet_input_t} is not a subclass of type nnet expects: {nnet_t.nnet_input_type()}"

        return None

    def __init__(self, domain: D, nnet_input_name: Tuple[str, str], nnet_name: str, nnet_kwargs: Dict[str, Any], **kwargs: Any):
        nnet_input_t: Type[NNetInput] = get_nnet_input_t(nnet_input_name)
        nnet_t: Type[DeepXubeNNet] = deepxube_nnet_factory.get_type(nnet_name)

        incompat_reason: Optional[str] = self.get_incompat_reason(domain, nnet_input_t, nnet_t)
        if incompat_reason is not None:
            raise TypeError(incompat_reason)

        self.domain: D = domain
        self.nnet_input_name: Tuple[str, str] = nnet_input_name
        self.nnet_name: str = nnet_name
        self.nnet_kwargs: Dict[str, Any] = nnet_kwargs

        super().__init__(**kwargs)

        self.nnet_input: Optional[NNInP] = None

    @abstractmethod
    def get_field_name(self) -> str:
        pass

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

    def __repr__(self) -> str:
        return f"NNetInputType: {get_nnet_input_t(self.nnet_input_name)} (name: {self.nnet_input_name})\n{super().__repr__()}"


H = TypeVar('H', bound=HeurFn)


class HeurNNetPar(DeepXubeNNetPar[H, CTX_T, D, NNInP, HeurNNet], ABC):
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


class HeurVNNetPar(HeurNNetPar[HeurVFn, None, Domain, StateGoalIn], ABC):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    @staticmethod
    def nnet_input_type() -> Type[StateGoalIn]:
        return StateGoalIn

    def get_field_name(self) -> str:
        return "heurv"

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


class HeurQNNetPar(HeurNNetPar[HeurQFn, CTX_T, D, NNInP], ABC):
    @abstractmethod
    def process_inputs(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> ProcessedInput[CTX_T]:
        pass

    def get_field_name(self) -> str:
        return "heurq"


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

    def get_field_name(self) -> str:
        return "policy"

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

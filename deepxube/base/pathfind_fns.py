from abc import abstractmethod, ABC
from dataclasses import dataclass, fields
from typing import List, Union, runtime_checkable, Protocol, Tuple, TypeVar, Generic, Type, Dict, Any, Optional, cast

import numpy as np
from numpy.typing import NDArray

from deepxube.utils.command_line_utils import get_name_args
from deepxube.utils import misc_utils
from deepxube.base.domain import State, Action, Goal, Domain
from deepxube.base.nnet import DeepXubeNNet, HeurNNet, PolicyNNet
from deepxube.base.nnet_input import NNetInput, StateGoalIn, PolicyNNetIn
from deepxube.factories.nnet_factory import deepxube_nnet_factory
from deepxube.factories.nnet_input_factory import get_nnet_input_t
from deepxube.pytorch.nnet_utils import NNetPar, NNF_T, PROCESSED_T, ProcessedInput


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


class DeepXubeNNetPar(NNetPar[NNF_T, PROCESSED_T], Generic[NNF_T, PROCESSED_T, D, NNInP, DXNNet]):
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
    def get_incompat_reason(cls, domain: Domain, nnet_input_t: Optional[Type[NNetInput]], nnet_t: Optional[Type[DeepXubeNNet]]) -> Optional[str]:
        if not isinstance(domain, cls.domain_type()):
            return f"Domain {domain} is not an instance of {cls.domain_type()}"
        elif (nnet_input_t is not None) and (not issubclass(nnet_input_t, cls.nnet_input_type())):
            return f"NNetInput type {nnet_input_t} is not a subclass of {cls.nnet_input_type()}"
        elif (nnet_t is not None) and (not issubclass(nnet_t, cls.nnet_type())):
            return f"DeepXubeNNet type {nnet_t} is not a subclass of {cls.nnet_type()}"
        elif (nnet_input_t is not None) and (nnet_t is not None) and (not issubclass(nnet_input_t, nnet_t.nnet_input_type())):
            return f"NNetInput type {nnet_input_t} is not a subclass of type nnet expects: {nnet_t.nnet_input_type()}"

        return None

    def __init__(self, domain: D, nnet_input_name: Optional[Tuple[str, str]], nnet_name_args: Optional[str], **kwargs: Any):

        nnet_input_t: Optional[Type[NNetInput]] = None
        nnet_t: Optional[Type[DeepXubeNNet]] = None
        if nnet_input_name is not None:
            nnet_input_t = get_nnet_input_t(nnet_input_name)
        if nnet_name_args is not None:
            nnet_t = deepxube_nnet_factory.get_type(get_name_args(nnet_name_args)[0])

        incompat_reason: Optional[str] = self.get_incompat_reason(domain, nnet_input_t, nnet_t)
        if incompat_reason is not None:
            raise TypeError(incompat_reason)

        self.domain: D = domain
        self.nnet_input_name: Optional[Tuple[str, str]] = nnet_input_name
        self.nnet_name_args: Optional[str] = nnet_name_args

        super().__init__(**kwargs)

        self.nnet_input: Optional[NNInP] = None

    @abstractmethod
    def get_field_name(self) -> str:
        pass

    def get_nnet(self) -> DXNNet:
        assert self.nnet_name_args is not None
        nnet_name, nnet_args = get_name_args(self.nnet_name_args)
        nnet_kwargs = deepxube_nnet_factory.get_kwargs(nnet_name, nnet_args)
        nnet_kwargs['nnet_input'] = self._get_nnet_input()

        self._add_nnet_kwargs(nnet_kwargs)
        nnet: DeepXubeNNet = deepxube_nnet_factory.build_class(nnet_name, nnet_kwargs)
        assert isinstance(nnet, self.nnet_type())
        return nnet

    @abstractmethod
    def _add_nnet_kwargs(self, nnet_kwargs: Dict) -> None:
        pass

    def _get_nnet_input(self) -> NNInP:
        if self.nnet_input is None:
            assert self.nnet_input_name is not None
            self.nnet_input = cast(NNInP, get_nnet_input_t(self.nnet_input_name)(domain=self.domain))

        assert self.nnet_input is not None
        return self.nnet_input

    def __getstate__(self) -> Dict:
        self.nnet_input = None
        return self.__dict__

    def __repr__(self) -> str:
        if self.nnet_input_name is None:
            return f"{type(self).__name__}()"
        else:
            assert self.nnet_name_args is not None
            return f"NNetInputType: {get_nnet_input_t(self.nnet_input_name)} (name: {self.nnet_input_name})\n{super().__repr__()}"


H = TypeVar('H', bound=HeurFn)


class HeurNNetPar(DeepXubeNNetPar[H, PROCESSED_T, D, NNInP, HeurNNet], ABC):
    @staticmethod
    def nnet_type() -> Type[HeurNNet]:
        return HeurNNet

    @abstractmethod
    def _qfix(self) -> bool:
        pass

    @abstractmethod
    def _out_dim(self) -> int:
        pass

    def _add_nnet_kwargs(self, nnet_kwargs: Dict) -> None:
        nnet_kwargs["q_fix"] = self._qfix()
        nnet_kwargs["out_dim"] = self._out_dim()


class HeurVNNetPar(HeurNNetPar[HeurVFn, None, Domain, StateGoalIn], ABC):
    @staticmethod
    def domain_type() -> Type[Domain]:
        return Domain

    @staticmethod
    def nnet_input_type() -> Type[StateGoalIn]:
        return StateGoalIn

    def get_default_fn(self) -> HeurVFn:
        class HeurZerosVFn(HeurVFn):
            def __call__(self, states: List[State], goals: List[Goal]) -> List[float]:
                return [0.0] * len(states)

        return HeurZerosVFn()

    def get_field_name(self) -> str:
        return "heurv"

    def process_inputs(self, states: List[State], goals: List[Goal]) -> ProcessedInput[None]:
        return ProcessedInput(self._get_nnet_input().to_np(states, goals), None)

    def process_outputs(self, outs: List[NDArray], processed: None) -> List[float]:
        heurs: NDArray = outs[0]
        heurs = np.maximum(heurs[:, 0], 0)
        return cast(List[float], heurs.astype(np.float64).tolist())

    def _qfix(self) -> bool:
        return False

    def _out_dim(self) -> int:
        return 1


class HeurQNNetPar(HeurNNetPar[HeurQFn, PROCESSED_T, D, NNInP], ABC):
    @abstractmethod
    def process_inputs(self, states: List[State], goals: List[Goal], actions_l: List[List[Action]]) -> ProcessedInput[PROCESSED_T]:
        pass

    def get_default_fn(self) -> HeurQFn:
        class HeurZerosQFn(HeurQFn):
            def __call__(self, states_in: List[State], goals_in: List[Goal], actions_l_in: List[List[Action]]) -> List[List[float]]:
                heur_vals_l: List[List[float]] = []
                for actions_in in actions_l_in:
                    heur_vals_l.append([0.0] * len(actions_in))
                return heur_vals_l

        return HeurZerosQFn()

    def get_field_name(self) -> str:
        return "heurq"


@dataclass(frozen=True)
class PolicyProcessed:
    num_states: int


def policy_fn_rand(domain: Domain, states: List[State], num_rand: int) -> Tuple[List[List[Action]], List[List[float]]]:
    if num_rand == 0:
        return [[] for _ in states], [[] for _ in states]

    states_rep: List[List[State]] = []
    for state in states:
        states_rep.append([state] * num_rand)

    states_rep_flat, split_idxs = misc_utils.flatten(states_rep)

    actions_samp_flat: List[Action] = domain.sample_state_action(states_rep_flat)
    actions_samp_l: List[List[Action]] = misc_utils.unflatten(actions_samp_flat, split_idxs)

    probs_l: List[List[float]] = []
    for actions_samp_i in actions_samp_l:
        probs_l.append([1.0 / len(actions_samp_i)] * len(actions_samp_i))

    return actions_samp_l, probs_l


class PolicyNNetPar(DeepXubeNNetPar[PolicyFn, PolicyProcessed, Domain, PolicyNNetIn, PolicyNNet]):
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

    def get_default_fn(self) -> PolicyFn:
        domain: Domain = self.domain
        num_samp: int = self.num_samp

        class PolicyFnRand(PolicyFn):
            def __call__(self, states: List[State], goals: List[Goal]) -> Tuple[List[List[Action]], List[List[float]]]:
                return policy_fn_rand(domain, states, num_samp)

        return PolicyFnRand()

    def get_field_name(self) -> str:
        return "policy"

    def to_np_train(self, states: List[State], goals: List[Goal], actions: List[Action]) -> List[NDArray[Any]]:
        return self._get_nnet_input().to_np(states, goals, actions)

    def process_inputs(self, states: List[State], goals: List[Goal]) -> ProcessedInput[PolicyProcessed]:
        return ProcessedInput(self._get_nnet_input().to_np_fn(states, goals), PolicyProcessed(len(states)))

    def process_outputs(self, outs: List[NDArray], processed: PolicyProcessed) -> Tuple[List[List[Action]], List[List[float]]]:
        actions_np: List[NDArray] = outs[0:-1]
        pdfs_np: NDArray = outs[-1]

        # assert dimensions match
        assert len(pdfs_np.shape) == 2
        assert pdfs_np.shape[0] == processed.num_states
        for actions_np_i in actions_np:
            assert actions_np_i.shape[0] == processed.num_states
            assert actions_np_i.shape[0] == pdfs_np.shape[0]
            assert actions_np_i.shape[1] == pdfs_np.shape[1]

        # convert to action object rep
        actions_l: List[List[Action]] = []
        pdfs_l: List[List[float]] = []
        for state_idx in range(processed.num_states):
            actions_np_state: List[NDArray[np.float64]] = [actions_np_i[state_idx] for actions_np_i in actions_np]
            pdfs_state: List[float] = pdfs_np[state_idx, :].tolist()

            actions_l.append(self._get_nnet_input().nnet_out_to_actions(actions_np_state))
            pdfs_l.append(pdfs_state)

        return actions_l, pdfs_l

    def _add_nnet_kwargs(self, nnet_kwargs: Dict) -> None:
        nnet_kwargs["num_samp"] = self.num_samp

    def __repr__(self) -> str:
        nnet: PolicyNNet = self.get_nnet()
        return f"{super().__repr__()}\n#Samp: {nnet.num_samp}"


@dataclass(frozen=True)
class UFNs:
    def get_field_names(self) -> List[str]:
        return [field.name for field in fields(self)]

    def get_up_fns(self) -> List[DeepXubeNNetPar]:
        return [self.get_up_fn(field_name) for field_name in self.get_field_names()]

    def get_up_fn(self, field_name: str) -> DeepXubeNNetPar:
        nnet_par: DeepXubeNNetPar = cast(DeepXubeNNetPar, getattr(self, field_name))
        assert isinstance(nnet_par, DeepXubeNNetPar)
        assert nnet_par.get_field_name() == field_name
        return nnet_par


@dataclass(frozen=True)
class UFNsHeurV(UFNs):
    heurv: HeurVNNetPar


@dataclass(frozen=True)
class UFNsHeurQ(UFNs):
    heurq: HeurQNNetPar


@dataclass(frozen=True)
class UFNsPolicy(UFNs):
    policy: PolicyNNetPar


@dataclass(frozen=True)
class UFNsHeurVPolicy(UFNsPolicy, UFNsHeurV):
    pass


@dataclass(frozen=True)
class UFNsHeurQPolicy(UFNsPolicy, UFNsHeurQ):
    pass

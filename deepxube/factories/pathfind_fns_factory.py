from typing import Any, Dict, Tuple, Type, List
from deepxube.utils.command_line_utils import get_name_args
from deepxube.factories.nnet_input_factory import get_domain_nnet_input_keys, get_nnet_input_t
from deepxube.factories.heuristic_factory import deepxube_nnet_factory
from deepxube.base.domain import Domain
from deepxube.base.nnet_input import NNetInput
from deepxube.base.pathfind_fns import PFNs, DeepXubeNNetPar
from deepxube.base.factory import FactoryAutoBuild, Factory

deepxube_nnet_par_factory: Factory[DeepXubeNNetPar] = Factory[DeepXubeNNetPar]("DeepXubeNNetPar")

pathfind_fns_factory: FactoryAutoBuild[PFNs] = FactoryAutoBuild[PFNs]("PathFindFNs")


def get_dx_nnet_par(domain: Domain, domain_name: str, nnet_par_name_args: str, nnet_name_args: str) -> Tuple[DeepXubeNNetPar, str]:
    nnet_par_name, nnet_par_args = get_name_args(nnet_par_name_args)
    nnet_name, nnet_args = get_name_args(nnet_name_args)

    # the types nnet_input must subclass
    nnet_input_t: Type[NNetInput] = deepxube_nnet_factory.get_type(nnet_name).nnet_input_type()
    nnet_par_input_t: Type[NNetInput] = deepxube_nnet_par_factory.get_type(nnet_par_name).nnet_input_type()

    # find nnet input and create class
    nnet_input_domain_names: List[Tuple[str, str]] = get_domain_nnet_input_keys(domain_name)
    for nnet_input_domain_name in nnet_input_domain_names:
        nnet_input_domain_t: Type[NNetInput] = get_nnet_input_t(nnet_input_domain_name)
        if issubclass(nnet_input_domain_t, nnet_input_t) and issubclass(nnet_input_domain_t, nnet_par_input_t):
            nnet_par_kwargs: Dict[str, Any] = deepxube_nnet_par_factory.get_kwargs(nnet_par_name, nnet_par_args)
            nnet_par_kwargs["domain"] = domain
            nnet_par_kwargs["nnet_input_name"] = nnet_input_domain_name
            nnet_par_kwargs["nnet_name"] = nnet_name
            nnet_par_kwargs["nnet_kwargs"] = deepxube_nnet_factory.get_kwargs(nnet_name, nnet_args)

            return deepxube_nnet_par_factory.build_class(nnet_par_name, nnet_par_kwargs), nnet_par_name

    raise ValueError(f"Cannot build fn for domain: {domain_name}, nnet_args: {nnet_name_args}, nnet_par_args: {nnet_par_name_args}, "
                     f"nnet_input type {nnet_input_t}, and nnet_par_type: {nnet_par_input_t}.\nNNet inputs checked: {nnet_input_domain_names}")

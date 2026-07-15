from typing import Any, Dict, Tuple, Type, List, Optional, Union

import torch
from torch import nn

from deepxube.utils.command_line_utils import get_name_args
from deepxube.pytorch.nnet_utils import NNetCallable, load_nnet
from deepxube.factories.nnet_input_factory import get_domain_nnet_input_keys, get_nnet_input_t
from deepxube.factories.nnet_factory import deepxube_nnet_factory
from deepxube.base.domain import Domain
from deepxube.base.nnet_input import NNetInput
from deepxube.base.nnet import DeepXubeNNet
from deepxube.base.pathfind_fns import PFNs, UFNs, DeepXubeNNetPar
from deepxube.base.factory import FactoryAutoBuild, Factory

deepxube_nnet_par_factory: Factory[DeepXubeNNetPar] = Factory[DeepXubeNNetPar]("DeepXubeNNetPar")

pathfind_fns_factory: FactoryAutoBuild[PFNs] = FactoryAutoBuild[PFNs]("PathFindFNs")

updater_fns_factory: FactoryAutoBuild[UFNs] = FactoryAutoBuild[UFNs]("UpdateFNs")


def get_dx_nnet_par(domain: Domain, domain_name: str, nnet_par_name_args: str, nnet_name_args: Optional[str]) -> Tuple[DeepXubeNNetPar, str]:
    # get nnet par type
    nnet_par_name, nnet_par_args = get_name_args(nnet_par_name_args)
    nnet_par_t: Type[DeepXubeNNetPar] = deepxube_nnet_par_factory.get_type(nnet_par_name)

    # get nnet type
    nnet_t: Optional[Type[DeepXubeNNet]] = None
    if nnet_name_args is not None:
        nnet_t = deepxube_nnet_factory.get_type(get_name_args(nnet_name_args)[0])

    # get possible nnet_input names for given domain
    nnet_input_domain_names: Union[List[Tuple[str, str]], List[None]]
    if nnet_t is None:
        nnet_input_domain_names = [None]
    else:
        nnet_input_domain_names = get_domain_nnet_input_keys(domain_name)

    # find nnet input and create class
    incompat_reasons: List[str] = []
    for nnet_input_domain_name in nnet_input_domain_names:
        nnet_input_domain_t: Optional[Type[NNetInput]] = None
        if nnet_input_domain_name is not None:
            nnet_input_domain_t = get_nnet_input_t(nnet_input_domain_name)

        incompat_reason: Optional[str] = nnet_par_t.get_incompat_reason(domain, nnet_input_domain_t, nnet_t)
        if incompat_reason is not None:
            incompat_reasons.append(incompat_reason + f" (NNetInput name: {nnet_input_domain_name})")
        else:
            nnet_par_kwargs: Dict[str, Any] = deepxube_nnet_par_factory.get_kwargs(nnet_par_name, nnet_par_args)
            nnet_par_kwargs["domain"] = domain
            nnet_par_kwargs["nnet_input_name"] = nnet_input_domain_name
            nnet_par_kwargs["nnet_name_args"] = nnet_name_args

            return deepxube_nnet_par_factory.build_class(nnet_par_name, nnet_par_kwargs), nnet_par_name

    incompat_reasons_str: str = '\n'.join(incompat_reasons)
    raise ValueError(f"Cannot build fn for domain: {domain_name}, nnet_args: {nnet_name_args}, nnet_par_args: {nnet_par_name_args}."
                     f"\nIncompatibility reasons:\n{incompat_reasons_str}")


def get_path_up_fns(domain: Domain, domain_name: str, fn_name_args_l: List[str], device: torch.device, nnet_files: Optional[List[Optional[str]]] = None,
                    nnet_batch_size: Optional[int] = None) -> Tuple[PFNs, UFNs]:
    nnet_fn_dict: Dict[str, NNetCallable] = dict()
    nnet_par_dict: Dict[str, DeepXubeNNetPar] = dict()
    if nnet_files is not None:
        assert len(nnet_files) == len(fn_name_args_l)

    for fn_idx, fn_arg in enumerate(fn_name_args_l):
        # get nnet par
        fn_arg_split: List[str] = fn_arg.split(",")
        nnet_name_args: Optional[str] = None
        if len(fn_arg_split) == 1:
            nnet_par_name_args = fn_arg_split[0]
        elif len(fn_arg_split) == 2:
            nnet_par_name_args, nnet_name_args = fn_arg_split[0], fn_arg_split[1]
        else:
            raise ValueError("Each element of fn_name_args_l must be either <fn> or <fn>,<nnet>")

        nnet_par, nnet_par_name = get_dx_nnet_par(domain, domain_name, nnet_par_name_args, nnet_name_args)

        # get fn
        fn: NNetCallable
        if nnet_name_args is not None:
            # get nnet
            nnet: nn.Module = nnet_par.get_nnet()

            # see if nnet file exists
            nnet_file: Optional[str] = None if (nnet_files is None) else nnet_files[fn_idx]

            # load nnet
            if nnet_file is not None:
                nnet_par.set_nnet_file(nnet_file)

                nnet = load_nnet(nnet_file, nnet_par.get_nnet())
                nnet.eval()
                nnet.to(device)
                nnet = nn.DataParallel(nnet)

            fn = nnet_par.get_nnet_fn(nnet, nnet_batch_size, device, None)
        else:
            fn = nnet_par.get_default_fn()

        field_name: str = nnet_par.get_field_name()
        nnet_fn_dict[field_name] = fn
        nnet_par_dict[field_name] = nnet_par

    pathfind_fns: PFNs = pathfind_fns_factory.build_class(nnet_fn_dict)
    updater_fns: UFNs = updater_fns_factory.build_class(nnet_par_dict)

    return pathfind_fns, updater_fns
